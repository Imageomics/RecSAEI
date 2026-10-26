"""Shared intervention results and SAE training provenance utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class InterventionResult:
    image: torch.Tensor
    original_tokens: torch.Tensor
    target_tokens: torch.Tensor
    mask: torch.Tensor
    details: dict[str, Any] = field(default_factory=dict)


def resolve_sae_training_metadata(shards_path: Path) -> tuple[dict[str, Any], Path]:
    """Resolve training-only token shuffles back to their spatial source shards."""
    metadata_path = shards_path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing shard metadata at {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shuffle_path = shards_path / "token_shuffle.json"
    if not shuffle_path.is_file():
        return metadata, shards_path

    shuffle = json.loads(shuffle_path.read_text(encoding="utf-8"))
    source_path = Path(str(shuffle.get("source_shards", ""))).resolve()
    source_metadata_path = source_path / "metadata.json"
    if not source_metadata_path.is_file():
        raise FileNotFoundError(
            f"token shuffle provenance points to missing source metadata at {source_metadata_path}"
        )
    source = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    invariant_keys = (
        "family",
        "ckpt",
        "layers",
        "d_model",
        "image_resize_mode",
        "encoder_implementation",
    )
    mismatches = [
        f"{key}={metadata.get(key)!r} (source {source.get(key)!r})"
        for key in invariant_keys
        if metadata.get(key) != source.get(key)
    ]
    expected_tokens = int(source.get("n_examples", 0)) * int(
        source.get("content_tokens_per_example", 0)
    )
    recorded_source_examples = shuffle.get("source_examples")
    recorded_source_tokens_per_example = shuffle.get(
        "source_content_tokens_per_example"
    )
    recorded_source_tokens = shuffle.get("source_tokens")
    recorded_output_tokens = shuffle.get("output_tokens")
    output_tokens = (
        expected_tokens
        if recorded_output_tokens is None
        else int(recorded_output_tokens)
    )
    provenance_checks = (
        ("source_examples", recorded_source_examples, source.get("n_examples")),
        (
            "source_content_tokens_per_example",
            recorded_source_tokens_per_example,
            source.get("content_tokens_per_example"),
        ),
        ("source_tokens", recorded_source_tokens, expected_tokens),
    )
    mismatches.extend(
        f"token_shuffle.{key}={actual!r} (source {expected!r})"
        for key, actual, expected in provenance_checks
        if actual is not None and actual != expected
    )
    if output_tokens <= 0 or output_tokens > expected_tokens:
        mismatches.append(
            f"token_shuffle.output_tokens={output_tokens!r} "
            f"(expected a value in [1, {expected_tokens}])"
        )
    recorded_max_tokens = shuffle.get("max_tokens")
    if recorded_max_tokens is not None and int(recorded_max_tokens) != output_tokens:
        mismatches.append(
            f"token_shuffle.max_tokens={recorded_max_tokens!r} "
            f"(output_tokens {output_tokens!r})"
        )
    if metadata.get("content_tokens_per_example") != 1:
        mismatches.append(
            "content_tokens_per_example="
            f"{metadata.get('content_tokens_per_example')!r} (expected 1 for token shuffle)"
        )
    if metadata.get("n_examples") != output_tokens:
        mismatches.append(
            f"n_examples={metadata.get('n_examples')!r} "
            f"(expected {output_tokens} shuffled tokens)"
        )
    if mismatches:
        raise ValueError("Invalid token-shuffle provenance: " + "; ".join(mismatches))
    return source, source_path


def validate_sae_provenance(checkpoint, *, backend, sae, token_mode: str) -> None:
    """Reject SAE weights trained on a different token representation.

    Training configs and shard metadata remain the provenance source. Shuffled
    patch shards are resolved back to their original spatial source metadata.
    """
    if token_mode not in ("content", "cls"):
        raise ValueError("token_mode must be content or cls")
    if token_mode == "cls" and not backend.model_has_cls_token:
        raise ValueError("This encoder has no CLS token")
    config = json.loads((Path(checkpoint).parent / "config.json").read_text())
    train = config["train_data"]
    metadata, _ = resolve_sae_training_metadata(Path(train["shards"]))
    layer = -1 if token_mode == "cls" else backend.editable_layer
    checks = {
        "tokens": (
            train.get("tokens"),
            "special" if token_mode == "cls" else "content",
        ),
        "family": (metadata.get("family"), backend.encoder_kind),
        "d_model": (metadata.get("d_model"), sae.cfg.d_model),
        "encoder_hidden_size": (sae.cfg.d_model, backend.rae.encoder.hidden_size),
        "content_tokens_per_example": (
            metadata.get("content_tokens_per_example"),
            backend.content_tokens_per_example,
        ),
        "image_resize_mode": (metadata.get("image_resize_mode"), "square"),
    }
    if backend.encoder_kind == "dinov3":
        from saev.data.dinov3 import OFFICIAL_ENCODER_IMPLEMENTATION

        filename = dinov3_encoder_filename(backend.stage1_params["encoder_name"])
        checks["ckpt"] = (Path(str(metadata.get("ckpt", ""))).name, filename)
        checks["encoder_implementation"] = (
            metadata.get("encoder_implementation"),
            OFFICIAL_ENCODER_IMPLEMENTATION,
        )
    else:
        from saev.data.siglip2 import HF_ENCODER_IMPLEMENTATION

        sizes = {"b": "base", "l": "large", "so400m": "so400m", "g": "giant-opt"}
        size = backend.stage1_params["encoder_name"].split("-")[-1]
        checks["ckpt"] = (
            metadata.get("ckpt"),
            f"google/siglip2-{sizes[size]}-patch16-256",
        )
        checks["encoder_implementation"] = (
            metadata.get("encoder_implementation"),
            HF_ENCODER_IMPLEMENTATION,
        )
    errors = [
        f"{key}={actual!r} (expected {expected!r})"
        for key, (actual, expected) in checks.items()
        if actual != expected
    ]
    if token_mode == "cls" and not metadata.get("cls_token"):
        errors.append("Source shards do not contain CLS tokens")

    def normalize(index):
        return backend.normalize_block_index(index, backend.num_blocks)

    if normalize(train["layer"]) != normalize(layer):
        errors.append(f"SAE layer {train['layer']} does not match expected {layer}")
    if normalize(layer) not in [
        normalize(index) for index in metadata.get("layers", [])
    ]:
        errors.append(f"Layer {layer} is absent from source shards")
    if errors:
        raise ValueError(
            "SAE provenance does not match the RAEv2 runtime: " + "; ".join(errors)
        )


def dinov3_encoder_filename(encoder_name: str) -> str:
    """Official LVD-1689M checkpoint matching the RAEv2 encoder configuration."""
    variants = {
        "s16": ("s16", "08c60483"),
        "s16plus": ("s16plus", "4057cbaa"),
        "b16": ("b16", "73cec8be"),
        "l16": ("l16", "8aa4cbdd"),
        "h16plus": ("h16plus", "7c1da9a5"),
        "7b16": ("7b16", "a955f4ea"),
    }
    variant, checksum = variants[encoder_name.split("-")[-1]]
    return f"dinov3_vit{variant}_pretrain_lvd1689m-{checksum}.pth"
