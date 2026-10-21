"""Verify the ported shuffle against exact seeded permutations on tiny shards."""

import json

import numpy as np
import pytest
import torch
from core.data.token_shuffle import run_token_shuffle
from core.intervention.common import resolve_sae_training_metadata
from saev.data.shards import Metadata, ShardInfo, ShardWriter


def write_source(tmp_path, *, cls_token=False, labels=True):
    root = tmp_path / "saev/shards"
    root.mkdir(parents=True)
    md = Metadata(
        family="dinov3",
        ckpt="test",
        layers=(-2, -1),
        content_tokens_per_example=2,
        cls_token=cls_token,
        d_model=3,
        n_examples=5,
        max_tokens_per_shard=12,
        data="test",
        dataset=tmp_path / "images",
        tag="source",
        image_resize_mode="square",
        encoder_implementation="test-official",
    )
    md.dump(root)
    content = np.arange(5 * 2 * 2 * 3, dtype=np.float32).reshape(5, 2, 2, 3)
    acts = content
    if cls_token:
        acts = np.concatenate(
            [np.full((5, 2, 1, 3), -99, dtype=np.float32), content], axis=2
        )
    patch_labels = np.arange(10, dtype=np.uint8).reshape(5, 2)
    with ShardWriter(root, md) as writer:
        writer.write_batch(torch.from_numpy(acts), 0, torch.from_numpy(patch_labels))
    if not labels:
        (root / "source/labels.bin").unlink()
    return root, md, content, patch_labels


def read_values(directory):
    md = Metadata.load(directory)
    values = []
    for shard in ShardInfo.load(directory):
        mapped = np.memmap(
            directory / shard.name, dtype=np.float32, mode="r", shape=md.shard_shape
        )
        values.append(np.array(mapped[: shard.n_examples]))
    return np.concatenate(values)


@pytest.mark.parametrize("cls_token", [False, True])
@pytest.mark.parametrize("max_tokens", [None, 7])
def test_shuffle_preserves_multilayer_content_labels_and_provenance(
    tmp_path, cls_token, max_tokens
):
    root, md, content, labels = write_source(tmp_path, cls_token=cls_token)
    source = root / "source"
    original = {path.name: path.read_bytes() for path in source.iterdir()}
    config = {
        "source_shards": source,
        "shards_root": root,
        "output_tag": "shuffled",
        "seed": 7,
        "source_chunk_examples": 1,
        "write_chunk_tokens": 3,
        "max_tokens": max_tokens,
    }
    result = run_token_shuffle(config)
    total = 10 if max_tokens is None else max_tokens
    rng = np.random.default_rng(7)
    order = (
        rng.permutation(10)
        if max_tokens is None
        else rng.choice(10, size=total, replace=False)
    )
    expected = content.transpose(0, 2, 1, 3).reshape(10, 2, 3)[order]
    np.testing.assert_array_equal(read_values(root / "shuffled")[:, :, 0], expected)
    output_labels = np.memmap(
        root / "shuffled/labels.bin", dtype=np.uint8, mode="r", shape=(total, 1)
    )
    np.testing.assert_array_equal(output_labels[:, 0], labels.reshape(-1)[order])
    shuffled_md = Metadata.load(root / "shuffled")
    assert shuffled_md.content_tokens_per_example == 1 and not shuffled_md.cls_token
    assert shuffled_md.n_examples == total
    assert shuffled_md.layers == md.layers
    assert shuffled_md.image_resize_mode == "square"
    assert result["output_tokens"] == total and result["source_tokens"] == 10
    restored, restored_path = resolve_sae_training_metadata(root / "shuffled")
    assert restored == json.loads((source / "metadata.json").read_text())
    assert restored_path == source
    assert original == {path.name: path.read_bytes() for path in source.iterdir()}
    second = run_token_shuffle(
        dict(config, output_tag="repeat", source_chunk_examples=3, write_chunk_tokens=2)
    )
    np.testing.assert_array_equal(
        read_values(root / "shuffled"), read_values(root / "repeat")
    )
    assert second["seed"] == 7


def test_shuffle_without_labels(tmp_path):
    root, _, content, _ = write_source(tmp_path, labels=False)
    result = run_token_shuffle(
        {"source_shards": root / "source", "output_tag": "shuffled"}
    )
    assert not result["has_labels"]
    assert (root / "shuffled/token_shuffle.json").is_file()
    expected = content.transpose(0, 2, 1, 3).reshape(10, 2, 3)[
        np.random.default_rng(42).permutation(10)
    ]
    np.testing.assert_array_equal(read_values(root / "shuffled")[:, :, 0], expected)


def test_shuffle_rejects_overwrite_and_shuffle_chains(tmp_path):
    root, _, _, _ = write_source(tmp_path)
    config = {"source_shards": root / "source", "output_tag": "shuffled"}
    run_token_shuffle(config)
    with pytest.raises(FileExistsError, match="overwrite"):
        run_token_shuffle(config)
    with pytest.raises(ValueError, match="image-ordered"):
        run_token_shuffle(
            dict(config, source_shards=root / "shuffled", output_tag="again")
        )
    assert not (root / "again").exists()


@pytest.mark.parametrize(
    "settings",
    [
        {"max_tokens": 0},
        {"max_tokens": 11},
        {"write_chunk_tokens": 0},
        {"source_chunk_examples": 0},
    ],
)
def test_invalid_shuffle_does_not_create_output(tmp_path, settings):
    root, _, _, _ = write_source(tmp_path)
    with pytest.raises(ValueError):
        run_token_shuffle(
            {"source_shards": root / "source", "output_tag": "invalid", **settings}
        )
    assert not (root / "invalid").exists()
