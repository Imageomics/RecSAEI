"""Materialize a globally shuffled, training-only token shard dataset."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from core.tools.io import dump_json
from tqdm import tqdm


def run_token_shuffle(config: dict[str, Any]) -> dict[str, Any]:
    """Write a globally permuted, one-content-token-per-example shard dataset.

    The result is intentionally only suitable for SAE training.  It preserves
    activations (and optional patch labels), but no longer preserves the mapping
    between an output row and an image or its other patches.
    """
    from saev import disk
    from saev.data import shards

    source_dir = Path(config["source_shards"]).resolve()
    shards_root = Path(config.get("shards_root", source_dir.parent)).resolve()
    output_tag = str(config["output_tag"])
    seed = int(config.get("seed", 42))
    max_tokens_value = config.get("max_tokens")
    write_chunk_tokens = int(config.get("write_chunk_tokens", 8192))
    source_chunk_examples = int(config.get("source_chunk_examples", 128))
    if write_chunk_tokens <= 0 or source_chunk_examples <= 0:
        raise ValueError(
            "write_chunk_tokens and source_chunk_examples must be positive"
        )
    if not disk.is_shards_root(shards_root):
        raise ValueError(
            f"shards_root must be an existing .../saev/shards directory: {shards_root}"
        )

    if (source_dir / "token_shuffle.json").exists():
        raise ValueError("Shuffle source must be image-ordered shards")
    source_md = shards.Metadata.load(source_dir)
    source_info = shards.ShardInfo.load(source_dir)
    source_info.validate(source_dir)
    if source_md.content_tokens_per_example <= 0:
        raise ValueError("Source shards have no content tokens to shuffle")
    source_labels_path = source_dir / "labels.bin"
    has_labels = source_labels_path.is_file()
    source_token_count = source_md.n_examples * source_md.content_tokens_per_example
    output_tokens = (
        source_token_count if max_tokens_value is None else int(max_tokens_value)
    )
    if not 0 < output_tokens <= source_token_count:
        raise ValueError(
            f"max_tokens must be between 1 and {source_token_count}, got {output_tokens}"
        )
    output_md = dataclasses.replace(
        source_md,
        content_tokens_per_example=1,
        cls_token=False,
        n_examples=output_tokens,
        tag=output_tag,
    )
    output_dir = shards_root / output_md.shard_id
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing shuffled shards: {output_dir}"
        )

    total_tokens = output_md.n_examples
    rng = np.random.default_rng(seed)
    selected_sources: np.ndarray | None = None
    selected_destinations: np.ndarray | None = None
    if total_tokens == source_token_count:
        permutation = rng.permutation(total_tokens)
        # `permutation` maps output positions to source positions. Its inverse
        # lets us scan the source sequentially and scatter into final positions.
        destination_for_source = np.empty_like(permutation)
        destination_for_source[permutation] = np.arange(
            total_tokens, dtype=permutation.dtype
        )
        del permutation
    else:
        # Sampling in output order provides both a uniform subset and a uniform
        # output permutation. Sort only the lookup copy so source shards can
        # still be read sequentially.
        selected_in_output_order = rng.choice(
            source_token_count, size=total_tokens, replace=False
        )
        selected_destinations = np.argsort(selected_in_output_order)
        selected_sources = selected_in_output_order[selected_destinations]
        del selected_in_output_order
        destination_for_source = None
    shuffled_activations = np.empty(
        (total_tokens, len(source_md.layers), source_md.d_model), dtype=np.float32
    )
    source_labels = (
        np.memmap(
            source_labels_path,
            mode="r",
            dtype=np.uint8,
            shape=(source_md.n_examples, source_md.content_tokens_per_example),
        )
        if has_labels
        else None
    )
    shuffled_labels = (
        np.empty((total_tokens, 1), dtype=np.uint8)
        if source_labels is not None
        else None
    )
    content_offset = int(source_md.cls_token)

    output_md.dump(shards_root)
    source_start = 0
    selected_count = 0
    for shard_index in tqdm(
        range(source_md.n_shards), desc="read-and-scatter", unit="shard"
    ):
        acts_path = source_dir / f"acts{shard_index:06}.bin"
        if not acts_path.is_file():
            raise FileNotFoundError(f"Missing source activation shard: {acts_path}")
        source_acts = np.memmap(
            acts_path, mode="r", dtype=np.float32, shape=source_md.shard_shape
        )
        n_examples = source_info[shard_index].n_examples
        for example_start in range(0, n_examples, source_chunk_examples):
            example_stop = min(example_start + source_chunk_examples, n_examples)
            count = (
                example_stop - example_start
            ) * source_md.content_tokens_per_example
            source_stop = source_start + count
            content = source_acts[
                example_start:example_stop,
                :,
                content_offset : content_offset + source_md.content_tokens_per_example,
                :,
            ]
            source_token_values = content.transpose(0, 2, 1, 3).reshape(
                count, len(source_md.layers), source_md.d_model
            )
            if selected_sources is None:
                assert destination_for_source is not None
                destinations = destination_for_source[source_start:source_stop]
                selected_offsets = slice(None)
                selected_count += count
            else:
                assert selected_destinations is not None
                selected_start = int(
                    np.searchsorted(selected_sources, source_start, side="left")
                )
                selected_stop = int(
                    np.searchsorted(selected_sources, source_stop, side="left")
                )
                source_positions = selected_sources[selected_start:selected_stop]
                selected_offsets = source_positions - source_start
                destinations = selected_destinations[selected_start:selected_stop]
                selected_count += len(source_positions)
            shuffled_activations[destinations] = source_token_values[selected_offsets]
            if source_labels is not None:
                assert shuffled_labels is not None
                label_example_start = (
                    source_start // source_md.content_tokens_per_example
                )
                label_example_stop = source_stop // source_md.content_tokens_per_example
                chunk_labels = source_labels[
                    label_example_start:label_example_stop
                ].reshape(-1)
                shuffled_labels[destinations, 0] = chunk_labels[selected_offsets]
            source_start = source_stop
        del source_acts
    if source_start != source_token_count:
        raise RuntimeError(
            f"Read {source_start} source content tokens, expected {source_token_count}"
        )
    if selected_count != total_tokens:
        raise RuntimeError(
            f"Selected {selected_count} content tokens, expected {total_tokens}"
        )
    del destination_for_source

    with shards.ShardWriter(shards_root, output_md) as writer:
        for output_start in tqdm(
            range(0, total_tokens, write_chunk_tokens),
            desc="write-shuffled-shards",
            unit="token-chunk",
        ):
            output_stop = min(output_start + write_chunk_tokens, total_tokens)
            activations = shuffled_activations[output_start:output_stop, :, None, :]
            labels = (
                None
                if shuffled_labels is None
                else shuffled_labels[output_start:output_stop]
            )
            writer.write_batch(
                torch.from_numpy(activations),
                output_start,
                None if labels is None else torch.from_numpy(labels),
            )

    output_info = shards.ShardInfo.load(output_dir)
    output_info.validate(output_dir)
    provenance = {
        "description": "True global permutation of source content-token activations; training-only layout with one token per example.",
        "source_shards": str(source_dir),
        "output_shards": str(output_dir),
        "seed": seed,
        "source_examples": source_md.n_examples,
        "source_content_tokens_per_example": source_md.content_tokens_per_example,
        "source_tokens": source_token_count,
        "output_tokens": total_tokens,
        "max_tokens": None if max_tokens_value is None else total_tokens,
        "write_chunk_tokens": write_chunk_tokens,
        "source_chunk_examples": source_chunk_examples,
        "activation_buffer_gib": round(shuffled_activations.nbytes / 2**30, 2),
        "has_labels": has_labels,
        "permutation": (
            "numpy.random.default_rng(seed).permutation(source_tokens)"
            if total_tokens == source_token_count
            else "numpy.random.default_rng(seed).choice(source_tokens, output_tokens, replace=False)"
        ),
        "implementation": "sequential source reads, in-memory scatter into global permutation, sequential output writes",
    }
    dump_json(provenance, output_dir / "token_shuffle.json")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize globally shuffled SAE content-token shards."
    )
    parser.add_argument("--source-shards", type=Path, required=True)
    parser.add_argument("--shards-root", type=Path)
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--source-chunk-examples", type=int, default=128)
    parser.add_argument("--write-chunk-tokens", type=int, default=8192)
    config = vars(parser.parse_args())
    if config["shards_root"] is None:
        config.pop("shards_root")
    print(json.dumps(run_token_shuffle(config), indent=2))


if __name__ == "__main__":
    main()
