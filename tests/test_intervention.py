"""Small differentiable models check steering without downloading checkpoints."""

import dataclasses
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from core.intervention import (
    CLSRequest,
    ExplicitTarget,
    GlobalMask,
    HighActivationMask,
    HighActivationMaskConfig,
    ManualPatchMask,
    PatchEditRequest,
    RAEv2Backend,
    ScaleTarget,
    intervene_cls,
    intervene_patch,
)
from core.intervention.cls import sae_gradient_direction
from core.intervention.common import (
    dinov3_encoder_filename,
    resolve_sae_training_metadata,
    validate_sae_provenance,
)
from PIL import Image
from torch import nn


class MixBlock(nn.Module):
    def forward(self, streams, *, rope):
        x = streams[0]
        return [x + x.mean(dim=1, keepdim=True) * rope]


class ToyEncoder(nn.Module):
    patch_size = 1
    hidden_size = 3

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.n_storage_tokens = 2
        self.model.blocks = nn.ModuleList([nn.Identity(), MixBlock()])
        self.model.norm = nn.LayerNorm(3, elementwise_affine=False)
        self.seen_inputs = []

    def forward(self, image):
        self.seen_inputs.append(image.detach().clone())
        patches = image.flatten(2).transpose(1, 2) / 255
        special = patches.new_full((1, 3, 3), 0.2)
        x = [torch.cat([special, patches], dim=1)]
        x = self.model.blocks[0](x)
        x = self.model.blocks[1](x, rope=0.25)
        return self.model.norm(x[0])[:, 3:]


class ToyRAE(nn.Module):
    resolution = 2
    do_normalization = True
    eps = 1e-5

    def __init__(self):
        super().__init__()
        self.encoder = ToyEncoder()
        self.latent_mean = torch.tensor(0.4)
        self.latent_var = torch.tensor(2.0)

    def decode(self, latents):
        raw = latents * torch.sqrt(self.latent_var + self.eps) + self.latent_mean
        return raw.sigmoid()


class ToySAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(d_model=3)
        self.W_dec = nn.Parameter(torch.eye(3))
        self.W_enc = nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        )
        self.bias = nn.Parameter(torch.tensor([0.01, 0.02, 0.03]))

    def encode(self, x):
        h = x @ self.W_enc
        return SimpleNamespace(f_x=h.relu(), h_x=h)

    def decode(self, codes):
        return (codes @ self.W_dec + self.bias)[:, None, :]


@pytest.fixture
def setup(tmp_path):
    backend = RAEv2Backend(
        rae=ToyRAE(), stage1_params={"encoder_name": "dinov3-vit-l16", "resolution": 2}
    )
    image = torch.tensor(
        [[[[0.1, 0.2], [0.3, 0.4]], [[0.3, 0.4], [0.5, 0.6]], [[0.5, 0.6], [0.7, 0.8]]]]
    )
    path = tmp_path / "input.png"
    pixels = (image[0].permute(1, 2, 0) * 255).round().to(torch.uint8).numpy()
    Image.fromarray(pixels).save(path)
    return backend, backend.load_image_tensor(path), ToySAE()


def test_patch_mask_and_residual(setup):
    backend, image, sae = setup
    result = intervene_patch(
        backend=backend,
        image_tensor=image,
        sae=sae,
        request=PatchEditRequest(
            ScaleTarget((0,), 2), "steering_delta", ManualPatchMask((1,))
        ),
    )
    assert result.mask.tolist() == [False, True, False, False]
    delta = result.target_tokens - result.original_tokens
    torch.testing.assert_close(delta[[0, 2, 3]], torch.zeros(3, 3))
    codes = sae.encode(result.original_tokens).f_x.detach()
    torch.testing.assert_close(delta[1], torch.tensor([codes[1, 0], 0, 0]))
    full = backend.encode_editable_tokens(image)
    edited = backend.reinsert_editable_tokens(full, result.target_tokens[None])
    torch.testing.assert_close(edited[:, :3], full[:, :3], rtol=0, atol=0)
    assert result.image.shape == (3, 2, 2)
    torch.testing.assert_close(backend.rae.encoder.seen_inputs[0], image)


@pytest.mark.parametrize(
    "mask",
    [
        GlobalMask(),
        HighActivationMask((0,), HighActivationMaskConfig(mode="top_k", top_k=1)),
        HighActivationMask((0,), HighActivationMaskConfig(mode="mass", min_patches=1)),
    ],
)
def test_patch_identity(setup, mask):
    backend, image, sae = setup
    result = intervene_patch(
        backend=backend,
        image_tensor=image,
        sae=sae,
        request=PatchEditRequest(ScaleTarget((0,), 1), "steering_delta", mask),
    )
    torch.testing.assert_close(
        result.target_tokens, result.original_tokens, rtol=0, atol=0
    )
    torch.testing.assert_close(
        result.image, backend.reconstruct_tensor(image)[0], rtol=0, atol=0
    )


def test_explicit_direct_decode(setup):
    backend, image, sae = setup
    result = intervene_patch(
        backend=backend,
        image_tensor=image,
        sae=sae,
        request=PatchEditRequest(
            ExplicitTarget((0,), {0: [2.0] * 4}), "direct_decode", GlobalMask()
        ),
    )
    codes = sae.encode(result.original_tokens).f_x.detach()
    codes[:, 0] = 2
    torch.testing.assert_close(result.target_tokens, sae.decode(codes)[:, -1])


@pytest.mark.parametrize("method", ["dense", "sae_gradient_delta"])
def test_cls_direction_and_zero_strength(setup, method):
    backend, image, sae = setup
    request = CLSRequest(
        0, 0, method=method, source_dimensions=3, min_patches=1, mass_fraction=0.5
    )
    zero = intervene_cls(
        backend=backend, image_tensor=image, cls_sae=sae, patch_sae=sae, request=request
    )
    torch.testing.assert_close(zero.target_tokens, zero.original_tokens, rtol=0, atol=0)
    torch.testing.assert_close(
        zero.image, backend.reconstruct_tensor(image)[0], rtol=0, atol=0
    )
    full = backend.capture_final_block_input(image)
    expected = sae.encode(backend.run_final_block(full, normalize=False)[:, 0]).f_x[
        0, 0
    ]
    assert zero.details["baseline_activation"] == pytest.approx(
        float(expected.detach())
    )
    for strength in (-0.1, 0.1):
        result = intervene_cls(
            backend=backend,
            image_tensor=image,
            cls_sae=sae,
            patch_sae=sae,
            request=dataclasses.replace(request, strength=strength),
        )
        assert (
            result.details["target_activation"] - zero.details["baseline_activation"]
        ) * strength > 0
        torch.testing.assert_close(
            result.target_tokens[~result.mask],
            result.original_tokens[~result.mask],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            (result.target_tokens - result.original_tokens).norm(),
            result.original_tokens.norm() * abs(strength),
        )
    assert all(
        not block._forward_hooks and not block._forward_pre_hooks
        for block in backend.backbone_layers
    )
    assert all(parameter.grad is None for parameter in sae.parameters())


def test_hook_cleanup_on_failure(setup, monkeypatch):
    backend, image, _ = setup

    def fail_forward(image):
        raise RuntimeError("Encoder failed")

    with monkeypatch.context() as patch:
        patch.setattr(backend.rae.encoder, "forward", fail_forward)
        for encode in (
            backend.encode_editable_tokens,
            backend.capture_final_block_input,
        ):
            with pytest.raises(RuntimeError, match="Encoder failed"):
                encode(image)
            assert all(
                not block._forward_hooks and not block._forward_pre_hooks
                for block in backend.backbone_layers
            )
    with pytest.raises(ValueError, match="shape"):
        backend.encode_target_tokens(
            image_tensor=image,
            editable_tokens=torch.ones(1, 8, 3),
            target_tokens=torch.ones(4, 3),
        )
    assert not backend.backbone_layers[0]._forward_hooks


def test_sae_ranking_and_sensitivity():
    codes = torch.tensor([[2.0, 0.0, 1.0], [1.0, 0.0, 2.0]])
    gradient = torch.tensor([[1.0, 10.0, -2.0], [1.0, 10.0, -2.0]])
    mask = torch.tensor([True, False])
    for mode, dims in [("activation_weighted", [0, 2]), ("sensitivity", [1, 2])]:
        delta, weights, selected = sae_gradient_direction(
            codes,
            gradient,
            torch.eye(3),
            mask,
            {"source_dimensions": 3, "selection_mode": mode},
        )
        assert [x["source_dimension"] for x in selected] == dims
        assert weights[1].count_nonzero() == 0
        torch.testing.assert_close(delta.norm(), torch.tensor(1.0))


def write_provenance(tmp_path, backend, *, tokens="content", layer=-2):
    from saev.data.dinov3 import OFFICIAL_ENCODER_IMPLEMENTATION

    shards = tmp_path / "shards"
    shards.mkdir()
    metadata = {
        "family": "dinov3",
        "d_model": 3,
        "content_tokens_per_example": 4,
        "image_resize_mode": "square",
        "encoder_implementation": OFFICIAL_ENCODER_IMPLEMENTATION,
        "ckpt": dinov3_encoder_filename("dinov3-vit-l16"),
        "layers": [-2, -1],
        "cls_token": True,
    }
    (shards / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"train_data": {"tokens": tokens, "layer": layer, "shards": str(shards)}}
        )
    )
    return tmp_path / "sae.pt", metadata, shards


def test_sae_provenance(setup, tmp_path):
    backend, _, sae = setup
    checkpoint, metadata, shards = write_provenance(tmp_path, backend)
    validate_sae_provenance(checkpoint, backend=backend, sae=sae, token_mode="content")
    metadata["image_resize_mode"] = "model_default"
    (shards / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="image_resize_mode"):
        validate_sae_provenance(
            checkpoint, backend=backend, sae=sae, token_mode="content"
        )


def test_cls_provenance(setup, tmp_path):
    backend, _, sae = setup
    checkpoint, _, _ = write_provenance(tmp_path, backend, tokens="special", layer=-1)
    validate_sae_provenance(checkpoint, backend=backend, sae=sae, token_mode="cls")
    with pytest.raises(ValueError, match="tokens"):
        validate_sae_provenance(
            checkpoint, backend=backend, sae=sae, token_mode="content"
        )


def test_token_shuffle_provenance(setup, tmp_path):
    backend, _, _ = setup
    _, metadata, source = write_provenance(tmp_path, backend)
    metadata.update(n_examples=2)
    (source / "metadata.json").write_text(json.dumps(metadata))
    shuffled = tmp_path / "shuffled"
    shuffled.mkdir()
    (shuffled / "metadata.json").write_text(
        json.dumps(dict(metadata, n_examples=8, content_tokens_per_example=1))
    )
    (shuffled / "token_shuffle.json").write_text(
        json.dumps({"source_shards": str(source), "output_tokens": 8})
    )
    actual, path = resolve_sae_training_metadata(shuffled)
    assert actual == metadata and path == source
    (shuffled / "metadata.json").write_text(
        json.dumps(
            dict(metadata, n_examples=8, content_tokens_per_example=1, layers=[-1])
        )
    )
    with pytest.raises(ValueError, match="layers"):
        resolve_sae_training_metadata(shuffled)


def test_from_config_local_assets_and_environment(tmp_path, monkeypatch):
    import yaml

    root = tmp_path / "RAEv2"
    stage = root / "src/stage1"
    stage.mkdir(parents=True)
    (stage / "__init__.py").write_text(
        "from test_intervention import ToyRAE\nimport os\nclass RAE(ToyRAE):\n    def __init__(self, **params):\n        assert 'DINOV3_REPO_DIR' not in os.environ\n        super().__init__()\n        self.params = params\n"
    )
    config_dir = root / "configs/decoder"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.json"
    config_file.write_text(json.dumps({"patch_size": "SHOULD BE RELOADED"}))
    for filename in ("decoder.pt", "stats.pt"):
        (root / filename).touch()
    ckpt_dir = root / "pretrained_models/encoders/dinov3"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / dinov3_encoder_filename("dinov3-vit-l16")).touch()
    params = {
        "encoder_name": "dinov3-vit-l16",
        "resolution": 2,
        "decoder_config_path": "configs/decoder",
        "pretrained_decoder_path": "decoder.pt",
        "normalization_stat_path": "stats.pt",
    }
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"stage_1": {"target": "stage1.RAE", "params": params}})
    )
    monkeypatch.setenv("DINOV3_REPO_DIR", "/old/repo")
    monkeypatch.setenv("DINOV3_CKPT_DIR", "/old/checkpoints")
    backend = RAEv2Backend.from_config(config, raev2_root=root)
    assert backend.rae.params["pretrained_decoder_path"] == str(root / "decoder.pt")
    compatible = json.loads(
        (Path(backend.rae.params["decoder_config_path"]) / "config.json").read_text()
    )
    assert compatible["patch_size"] == 16
    assert json.loads(config_file.read_text())["patch_size"] == "SHOULD BE RELOADED"
    assert os.environ["DINOV3_REPO_DIR"] == "/old/repo"
    assert os.environ["DINOV3_CKPT_DIR"] == "/old/checkpoints"
    params["encoder_name"] = "dinov3mls-vit-l16[layers=21.23]"
    config.write_text(
        yaml.safe_dump({"stage_1": {"target": "stage1.RAE", "params": params}})
    )
    with pytest.raises(ValueError, match="K1"):
        RAEv2Backend.from_config(config, raev2_root=root)


def test_real_saev_checkpoint_load_and_edit(setup, tmp_path):
    from saev.nn import modeling

    backend, image, _ = setup
    checkpoint, _, _ = write_provenance(tmp_path, backend)
    sae = modeling.SparseAutoencoder(
        modeling.SparseAutoencoderConfig(
            d_model=3,
            d_sae=4,
            activation=modeling.TopK(top_k=2),
        )
    ).eval()
    modeling.dump(checkpoint, sae)
    loaded = backend.load_sae(checkpoint)
    original = backend.select_editable_tokens(backend.encode_editable_tokens(image))[0]
    torch.testing.assert_close(loaded.encode(original).f_x, sae.encode(original).f_x)
    result = intervene_patch(
        backend=backend,
        sae=loaded,
        image_tensor=image,
        request=PatchEditRequest(ScaleTarget((0,), 1), "steering_delta", GlobalMask()),
    )
    torch.testing.assert_close(
        result.image, backend.reconstruct_tensor(image)[0], atol=0, rtol=0
    )


def test_siglip_patch_backend(setup):
    backend, image, sae = setup

    class SiglipBlock(nn.Module):
        def forward(self, x):
            return (x + 0.1,)

    class SiglipEncoder(nn.Module):
        patch_size = 1
        hidden_size = 3

        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.vision_model = nn.Module()
            self.model.vision_model.encoder = nn.Module()
            self.model.vision_model.encoder.layers = nn.ModuleList(
                [SiglipBlock(), SiglipBlock()]
            )

        def forward(self, x):
            tokens = x.flatten(2).transpose(1, 2) / 255
            for layer in self.model.vision_model.encoder.layers:
                tokens = layer(tokens)[0]
            return tokens

    rae = ToyRAE()
    rae.encoder = SiglipEncoder()
    backend = RAEv2Backend(
        rae=rae, stage1_params={"encoder_name": "siglip2-vit-b", "resolution": 2}
    )
    assert backend.patch_slice == slice(0, 4)
    result = intervene_patch(
        backend=backend,
        sae=sae,
        image_tensor=image,
        request=PatchEditRequest(ScaleTarget((0,), 1), "steering_delta", GlobalMask()),
    )
    torch.testing.assert_close(
        result.image, backend.reconstruct_tensor(image)[0], atol=0, rtol=0
    )
    with pytest.raises(ValueError, match="CLS tokens"):
        intervene_cls(
            backend=backend,
            cls_sae=sae,
            image_tensor=image,
            request=CLSRequest(0, 1, method="dense"),
        )


@pytest.mark.parametrize("method", ["dense", "sae_gradient_delta"])
def test_cls_strength_uses_all_original_patches(setup, method):
    backend, image, sae = setup
    references = []
    for fraction in (0.25, 1.0):
        result = intervene_cls(
            backend=backend,
            image_tensor=image,
            cls_sae=sae,
            patch_sae=sae,
            request=CLSRequest(0, 0.1, method=method, mass_fraction=fraction),
        )
        original = result.original_tokens
        delta = result.target_tokens - original
        expected = 0.1 * original.square().sum().sqrt()
        torch.testing.assert_close(delta.norm(), expected)
        assert result.details["strength_reference"] == "all_original_patch_tokens"
        references.append((result.mask.sum().item(), delta.norm()))
        if fraction < 1:
            assert result.mask.sum() < len(original)
            assert not torch.isclose(expected, 0.1 * original[result.mask].norm())
            patch_norm_sum = original[result.mask].norm(dim=-1).sum()
            patch_norm_mean = original[result.mask].norm(dim=-1).mean()
            assert not torch.isclose(expected, 0.1 * patch_norm_sum)
            assert not torch.isclose(expected, 0.1 * patch_norm_mean)
    assert references[0][0] < references[1][0]
    torch.testing.assert_close(references[0][1], references[1][1])


def test_cls_defaults_and_removed_method(setup):
    backend, image, sae = setup
    assert CLSRequest(0, 0).method == "sae_gradient_delta"
    with pytest.raises(ValueError, match="Unknown CLS steering method"):
        intervene_cls(
            backend=backend,
            image_tensor=image,
            cls_sae=sae,
            patch_sae=sae,
            request=CLSRequest(0, 0, method="sae"),
        )


@pytest.mark.parametrize("color", [(0, 128, 255), (1, 1, 1)])
def test_image_loader_preserves_pixel_scale(setup, tmp_path, color):
    backend, _, _ = setup
    path = tmp_path / "input.png"
    Image.new("RGB", (2, 2), color).save(path)
    loaded = backend.load_image_tensor(path)
    assert loaded.shape == (1, 3, 2, 2)
    assert loaded.dtype == torch.float32
    torch.testing.assert_close(
        loaded[0, :, 0, 0], torch.tensor(color, dtype=torch.float32)
    )
    backend.encode_decoder_tokens(loaded)
    torch.testing.assert_close(
        backend.rae.encoder.seen_inputs[-1], loaded, atol=0, rtol=0
    )
