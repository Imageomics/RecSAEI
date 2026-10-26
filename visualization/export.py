"""Export current saev inference results for patch/CLS feature browsing.

Run from the repository root: python -m visualization.export --config <yaml>.
Relative paths in the YAML are relative to the working directory.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml
from core.selection.activation import pool_activations
from core.selection.class_specific import fit_class_scores
from scipy import sparse
from torchvision.datasets import ImageFolder


def load_source(source, mode):
    from saev.data.shards import Metadata

    shards = Path(source["shards_dir"])
    metadata = json.loads((shards / "metadata.json").read_text())
    if (shards / "token_shuffle.json").exists():
        raise ValueError("Viewer exports require ordered image shards")
    if (
        metadata.get("image_resize_mode") != "square"
        or metadata.get("family") != "dinov3"
    ):
        raise ValueError("Viewer currently requires RAEv2 DINOv3 square preprocessing")
    if mode == "cls" and not metadata.get("cls_token"):
        raise ValueError("CLS export requires CLS tokens in the source shards")
    inference = source.get("inference_dir")
    if not inference:
        inference = (
            Path(source["run_dir"]) / "inference" / Metadata.load(shards).shard_id
        )
    train = json.loads(
        (Path(source["run_dir"]) / "checkpoint/config.json").read_text()
    )["train_data"]
    expected = "special" if mode == "cls" else "content"
    if train["tokens"] != expected:
        raise ValueError(f"SAE training token mode must be {expected}")
    if train["layer"] not in metadata["layers"]:
        raise ValueError("SAE layer is absent from inference shards")
    dataset = ImageFolder(str(metadata["dataset"]))
    tokens = 1 if mode == "cls" else metadata["content_tokens_per_example"]
    codes = sparse.load_npz(Path(inference) / "token_acts.npz").tocsr()
    if (
        len(dataset) != metadata["n_examples"]
        or codes.shape[0] != len(dataset) * tokens
    ):
        raise ValueError("Inference rows do not match the image/token layout")
    if not np.isfinite(codes.data).all():
        raise ValueError("Inference activations must be finite")
    # Pool one image at a time; never densify the whole token activation matrix.
    pooled = {
        name: sparse.vstack(
            [
                sparse.csr_array(
                    pool_activations(
                        codes[i * tokens : (i + 1) * tokens], pooling=name
                    )[None]
                )
                for i in range(len(dataset))
            ],
            format="csr",
        )
        for name in ("mean", "max")
    }
    images = [
        {
            "image_id": str(i),
            "source_path": str(Path(path).resolve()),
            "class_name": dataset.classes[label],
            "class_id": label,
        }
        for i, (path, label) in enumerate(dataset.samples)
    ]
    return metadata, codes, pooled, images


def export_viewer(config):
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "title": config.get("title", "SAE Feature Viewer"),
        "modes": {},
        "images": "images.json",
    }
    reference_images = None
    for mode, source in config["sources"].items():
        if mode not in ("patch", "cls"):
            raise ValueError("Source modes must be patch or cls")
        metadata, codes, pooled, images = load_source(source, mode)
        if reference_images is not None and images != reference_images:
            raise ValueError("Patch and CLS sources must have the same ordered images")
        reference_images = images
        destination = root / mode
        destination.mkdir(exist_ok=True)
        sparse.save_npz(destination / "token_acts.npz", codes)
        for pooling, matrix in pooled.items():
            sparse.save_npz(destination / f"{pooling}.npz", matrix)
        tokens = metadata["content_tokens_per_example"]
        side = math.isqrt(tokens)
        if side * side != tokens:
            raise ValueError("Expected a square encoder patch grid")
        information = {
            "d_sae": codes.shape[1],
            "tokens_per_image": 1 if mode == "cls" else tokens,
            "image_size": side * 16,
            "class_selection": False,
            "sae_checkpoint": str(
                (Path(source["run_dir"]) / "checkpoint/sae.pt").resolve()
            ),
            "layer": json.loads(
                (Path(source["run_dir"]) / "checkpoint/config.json").read_text()
            )["train_data"]["layer"],
            "encoder_implementation": metadata.get("encoder_implementation"),
            "ranking_scope": "all exported inference images",
        }
        if source.get("fit"):
            fit_metadata, _, fit_pooled, fit_images = load_source(source["fit"], mode)
            keys = (
                "family",
                "ckpt",
                "encoder_implementation",
                "content_tokens_per_example",
                "d_model",
            )
            if (
                any(metadata.get(key) != fit_metadata.get(key) for key in keys)
                or fit_pooled["max"].shape[1] != codes.shape[1]
            ):
                raise ValueError(
                    "Class-fit and browsing sources must use the same encoder and SAE width"
                )
            if (
                Path(source["fit"]["run_dir"]).resolve()
                != Path(source["run_dir"]).resolve()
            ):
                raise ValueError(
                    "Class-fit and browsing sources must use the same SAE run"
                )
            scores = fit_class_scores(
                fit_pooled["max"], [row["class_name"] for row in fit_images]
            )
            np.savez_compressed(destination / "classes.npz", **scores)
            information["class_selection"] = True
        manifest["modes"][mode] = information
    if not manifest["modes"]:
        raise ValueError("At least one source is required")
    (root / "images.json").write_text(json.dumps({"items": reference_images}, indent=2))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    export_viewer(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
