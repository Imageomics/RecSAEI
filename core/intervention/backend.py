"""One RAEv2 backend for raw encoder tokens and decoder reconstruction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


class RAEv2Backend:
    """Image tensors must come from load_image_tensor().

    Token indices always select patches and skip internal CLS/register tokens.
    CLS interventions use a separate target SAE on raw final-block CLS output.
    Hooks and replay preserve the encoder's auxiliary arguments (including RoPE).
    """

    def __init__(
        self,
        *,
        rae: torch.nn.Module,
        stage1_params: Mapping[str, Any],
        device: str | torch.device = "cpu",
        editable_layer: int = -2,
    ):
        self.device = torch.device(device)
        self.rae = rae.to(self.device).eval()
        self.stage1_params = dict(stage1_params)
        self.input_size = int(
            self.stage1_params.get("resolution", getattr(rae, "resolution", 256))
        )
        self.editable_layer = editable_layer
        self.backbone = self._resolve_backbone().to(self.device).eval()
        self.backbone_layers = self._get_backbone_layers()
        self.num_blocks = len(self.backbone_layers)
        self.block_index = self.normalize_block_index(editable_layer, self.num_blocks)
        self.num_register_tokens = self._get_num_register_tokens()
        self.model_has_cls_token = self.encoder_kind != "siglip2"
        if (
            self.patch_size <= 0
            or self.input_size <= 0
            or self.input_size % self.patch_size
        ):
            raise ValueError(
                "Resolution must be a positive multiple of encoder patch size"
            )
        self.content_tokens_per_example = (self.input_size // self.patch_size) ** 2
        start = self.num_register_tokens + int(self.model_has_cls_token)
        self.patch_slice = slice(start, start + self.content_tokens_per_example)
        self.token_indices = torch.arange(start, self.patch_slice.stop)

    def reconstruct_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return self.decode_decoder_tokens(
            self.encode_decoder_tokens(image_tensor)
        ).unsqueeze(0)

    def capture_final_block_input(self, image_tensor: torch.Tensor) -> torch.Tensor:
        captured = []

        def capture(module, args, kwargs):
            captured.append((args, kwargs))

        handle = self.backbone_layers[-1].register_forward_pre_hook(
            capture, with_kwargs=True
        )
        try:
            with torch.no_grad():
                self.rae.encoder(image_tensor.to(self.device))
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(
                f"Expected one final-block input capture, got {len(captured)}"
            )
        args, kwargs = captured[0]
        self._final_block_aux_args = args[1:]
        self._final_block_aux_kwargs = kwargs
        self._final_block_input_is_list = isinstance(args[0], list)
        return self.unwrap_hidden_states(args[0]).detach().requires_grad_(True)

    def run_final_block(
        self, final_block_input: torch.Tensor, *, normalize: bool = True
    ) -> torch.Tensor:
        if not hasattr(self, "_final_block_aux_args"):
            raise RuntimeError(
                "Call capture_final_block_input() before replaying the final block"
            )
        block_input = (
            [final_block_input]
            if self._final_block_input_is_list
            else final_block_input
        )
        hidden = self.unwrap_hidden_states(
            self.backbone_layers[-1](
                block_input, *self._final_block_aux_args, **self._final_block_aux_kwargs
            )
        )
        if normalize:
            for name in ("layernorm", "norm", "post_layernorm"):
                if hasattr(self.backbone, name):
                    hidden = getattr(self.backbone, name)(hidden)
                    break
        return hidden

    def _get_backbone_layers(self) -> list[torch.nn.Module]:
        layers = getattr(getattr(self.backbone, "encoder", None), "layer", None)
        if layers is None:
            layers = getattr(getattr(self.backbone, "encoder", None), "layers", None)
        if layers is None:
            layers = getattr(self.backbone, "blocks", None)
        if layers is None:
            raise AttributeError(
                f"Could not find transformer layers on live-edit backbone: {type(self.backbone).__name__}"
            )
        return list(layers)

    def _get_num_register_tokens(self) -> int:
        if self.encoder_kind == "dinov3":
            return int(self.backbone.n_storage_tokens)
        return 0  # SigLIP 2 has no register tokens.

    @staticmethod
    def unwrap_hidden_states(output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return RAEv2Backend.unwrap_hidden_states(output[0])
        if isinstance(output, list):
            if len(output) != 1:
                raise ValueError(
                    f"Expected one DINOv3 crop/token stream, got {len(output)}"
                )
            return RAEv2Backend.unwrap_hidden_states(output[0])
        if hasattr(output, "last_hidden_state"):
            output = output.last_hidden_state
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Expected hidden-state tensor, got {type(output)}")
        return output

    @staticmethod
    def replace_hidden_states(output: Any, replacement: torch.Tensor) -> Any:
        hidden = RAEv2Backend.unwrap_hidden_states(output)
        value = replacement.to(dtype=hidden.dtype, device=hidden.device)
        if isinstance(output, list):
            return [value]
        if isinstance(output, tuple):
            return (
                RAEv2Backend.replace_hidden_states(output[0], replacement),
                *output[1:],
            )
        return value

    @staticmethod
    def normalize_block_index(index: int, n_blocks: int) -> int:
        normalized = index if index >= 0 else n_blocks + index
        if normalized < 0 or normalized >= n_blocks:
            raise ValueError(
                f"editable_layer={index} is out of range for {n_blocks} blocks"
            )
        return normalized

    def load_image_tensor(self, image_path: str | Path) -> torch.Tensor:
        """Load one RGB image as a CPU float32 tensor [1, 3, H, W] in [0, 255]."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")
        with Image.open(path) as image:
            return pil_to_tensor(image.convert("RGB")).to(torch.float32).unsqueeze(0)

    def select_editable_tokens(self, full_tokens: torch.Tensor) -> torch.Tensor:
        return full_tokens[:, self.token_indices.to(full_tokens.device), :]

    def reinsert_editable_tokens(
        self, full_tokens: torch.Tensor, edited_tokens: torch.Tensor
    ) -> torch.Tensor:
        output = full_tokens.clone()
        output[:, self.token_indices.to(full_tokens.device), :] = edited_tokens
        return output

    def normalize_for_decode(self, final_patch_tokens: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, dim = final_patch_tokens.shape
        side = math.isqrt(n_tokens)
        if side * side != n_tokens:
            raise ValueError("Decoder tokens must form a square patch grid")
        latents = (
            final_patch_tokens.transpose(1, 2).contiguous().view(bsz, dim, side, side)
        )
        if self.rae.do_normalization:
            latent_mean = (
                self.rae.latent_mean.to(latents.device)
                if self.rae.latent_mean is not None
                else 0
            )
            latent_var = (
                self.rae.latent_var.to(latents.device)
                if self.rae.latent_var is not None
                else 1
            )
            latents = (latents - latent_mean) / torch.sqrt(latent_var + self.rae.eps)
        return latents

    @property
    def patch_size(self) -> int:
        return int(
            getattr(
                self.rae.encoder,
                "patch_size",
                getattr(self.rae, "encoder_patch_size", 0),
            )
        )

    def encode_decoder_tokens(self, image_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.rae.encoder(image_tensor.to(self.device))

    @torch.no_grad()
    def decode_decoder_tokens(self, decoder_tokens: torch.Tensor) -> torch.Tensor:
        latents = self.normalize_for_decode(decoder_tokens.to(self.device))
        return self.rae.decode(latents).clamp(0, 1).detach().cpu()[0]

    def encode_target_tokens(
        self,
        *,
        image_tensor: torch.Tensor,
        editable_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        edited_full = self.reinsert_editable_tokens(
            editable_tokens, target_tokens.unsqueeze(0)
        )
        replacement = edited_full.to(self.device)
        calls = 0

        def replace_hook(
            module: torch.nn.Module, args: tuple[Any, ...], output: Any
        ) -> Any:
            nonlocal calls
            calls += 1
            hidden = self.unwrap_hidden_states(output)
            if hidden.shape != replacement.shape:
                raise ValueError(
                    f"Edited {self.encoder_kind} token shape {tuple(replacement.shape)} "
                    f"does not match block output {tuple(hidden.shape)}"
                )
            return self.replace_hidden_states(output, replacement)

        handle = self.backbone_layers[self.block_index].register_forward_hook(
            replace_hook
        )
        try:
            with torch.no_grad():
                output = self.rae.encoder(image_tensor.to(self.device))
        finally:
            handle.remove()
        if calls != 1:
            raise RuntimeError(
                f"Expected one {self.encoder_kind} replacement at block "
                f"{self.block_index}, got {calls}"
            )
        return output

    def reencode_selected_tokens(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a CHW image from load_image_tensor()[0]."""
        with torch.no_grad():
            editable = self.encode_editable_tokens(image.unsqueeze(0))
            return self.select_editable_tokens(editable).squeeze(0)

    @property
    def encoder_kind(self) -> str:
        return self._encoder_kind

    def _resolve_backbone(self) -> torch.nn.Module:
        encoder_name = str(self.stage1_params.get("encoder_name", ""))
        if encoder_name.startswith("dinov3-"):
            self._encoder_kind = "dinov3"
        elif encoder_name.startswith("siglip2-"):
            self._encoder_kind = "siglip2"
        else:
            raise AttributeError(
                f"RAEv2 reconstruction supports encoder '{encoder_name}', but live "
                "SAE editing currently requires DINOv3 or SigLIP 2"
            )
        if hasattr(self.rae.encoder, "model"):
            return getattr(
                self.rae.encoder.model, "vision_model", self.rae.encoder.model
            )
        raise AttributeError(
            f"RAEv2 encoder wrapper for '{encoder_name}' does not expose its model"
        )

    def encode_editable_tokens(self, image_tensor: torch.Tensor) -> torch.Tensor:
        captured: list[torch.Tensor] = []

        def capture_hook(
            module: torch.nn.Module, args: tuple[Any, ...], output: Any
        ) -> None:
            captured.append(self.unwrap_hidden_states(output))

        handle = self.backbone_layers[self.block_index].register_forward_hook(
            capture_hook
        )
        try:
            with torch.no_grad():
                self.rae.encoder(image_tensor.to(self.device))
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(
                f"Expected one {self.encoder_kind} activation capture at block "
                f"{self.block_index}, got {len(captured)}"
            )
        return captured[0]

    def decode_target_tokens(
        self,
        *,
        image_tensor: torch.Tensor,
        editable_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        final_tokens = self.encode_target_tokens(
            image_tensor=image_tensor,
            editable_tokens=editable_tokens,
            target_tokens=target_tokens,
        )
        return self.decode_decoder_tokens(final_tokens)

    @classmethod
    def from_config(
        cls,
        config: str | Path,
        *,
        device: str | torch.device = "cpu",
        editable_layer: int = -2,
        raev2_root: str | Path | None = None,
        dinov3_checkpoint_dir: str | Path | None = None,
    ) -> RAEv2Backend:
        """Load a released K1 stage_1 YAML using locally prepared RAEv2 assets.

        Asset paths in the YAML are relative to raev2_root (default: vendored
        RAEv2). Encoder loading uses the official implementation pinned there.
        """
        import yaml

        root = Path(
            raev2_root or Path(__file__).resolve().parents[2] / "RAEv2"
        ).resolve()
        payload = yaml.safe_load(Path(config).read_text())
        stage = payload.get("stage_1", {})
        if stage.get("target") != "stage1.RAE":
            raise ValueError("Expected a K1 stage_1 config with target stage1.RAE")
        params = dict(stage["params"])
        if not params.get("encoder_name", "").startswith(("dinov3-", "siglip2-")):
            raise ValueError("Only K1 DINOv3 and SigLIP 2 encoders are supported")
        for key in (
            "decoder_config_path",
            "pretrained_decoder_path",
            "normalization_stat_path",
        ):
            if not params.get(key):
                raise ValueError(f"Stage-1 config requires {key}")
            path = Path(params[key]).expanduser()
            path = path if path.is_absolute() else root / path
            if not path.exists():
                raise FileNotFoundError(path)
            params[key] = str(path.resolve())
        config_dir = Path(params["decoder_config_path"])
        decoder_config = json.loads((config_dir / "config.json").read_text())
        if decoder_config.get("patch_size") == "SHOULD BE RELOADED":
            decoder_config["patch_size"] = int(params.get("decoder_patch_size", 16))
            encoded = json.dumps(decoder_config, sort_keys=True)
            digest = hashlib.sha256(encoded.encode()).hexdigest()[:16]
            destination = (
                Path(tempfile.gettempdir()) / "recsaei-decoder-configs" / digest
            )
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "config.json").write_text(encoded)
            params["decoder_config_path"] = str(destination)
        src = root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        namespace = (
            "_recsaei_raev2_" + hashlib.sha256(str(root).encode()).hexdigest()[:12]
        )
        if namespace not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                namespace,
                src / "stage1" / "__init__.py",
                submodule_search_locations=[str(src / "stage1")],
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load RAEv2 at {root}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[namespace] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(namespace, None)
                raise
        if params["encoder_name"].startswith("dinov3-"):
            from .common import dinov3_encoder_filename

            checkpoint_dir = Path(
                dinov3_checkpoint_dir or root / "pretrained_models/encoders/dinov3"
            )
            checkpoint = checkpoint_dir / dinov3_encoder_filename(
                params["encoder_name"]
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
        environment = {
            "DINOV3_CKPT_DIR": str(Path(dinov3_checkpoint_dir).resolve())
            if dinov3_checkpoint_dir
            else str(root / "pretrained_models/encoders/dinov3"),
            "DINOV3_REPO_DIR": None,
        }
        previous = {key: os.environ.get(key) for key in environment}
        try:
            for key, value in environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            rae = sys.modules[namespace].RAE(**params)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        return cls(
            rae=rae, stage1_params=params, device=device, editable_layer=editable_layer
        )

    def load_sae(
        self, checkpoint: str | Path, *, token_mode: str = "content"
    ) -> torch.nn.Module:
        """Load an SAE and check its saved token, layer and encoder provenance."""
        from saev.nn import modeling

        from .common import validate_sae_provenance

        sae = (
            modeling.load(Path(checkpoint), device=str(self.device))
            .to(self.device)
            .eval()
        )
        validate_sae_provenance(
            checkpoint, backend=self, sae=sae, token_mode=token_mode
        )
        return sae
