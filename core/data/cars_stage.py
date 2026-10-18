"""Stage Stanford Cars under all/ using the shared dataset pipeline."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from .common import DatasetAdapter, ImageRecord, StageManifest, validated_relative_path


def _mat_items(value: Any) -> list[Any]:
    return np.atleast_1d(value).ravel().tolist()


def _load_class_names(path: Path) -> list[str]:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "class_names" not in payload:
        raise ValueError(f"Stanford Cars metadata has no class_names variable: {path}")
    names = [str(value) for value in _mat_items(payload["class_names"])]
    if not names or any(not name for name in names):
        raise ValueError("Stanford Cars class names must be non-empty")
    return names


def _load_annotation_structs(path: Path) -> list[Any]:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "annotations" not in payload:
        raise ValueError(f"Stanford Cars annotation file has no annotations: {path}")
    annotations = _mat_items(payload["annotations"])
    if not annotations:
        raise ValueError(f"Stanford Cars annotation file is empty: {path}")
    return annotations


def _annotation_int(value: Any) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"Stanford Cars annotation must be integer-valued: {value}")
    return int(number)


class CarsAdapter(DatasetAdapter):
    """Stage Cars under all/ while retaining available source class labels.

    Output under staged_root::

        train/all/<filename>  # official training split
        val/all/<filename>    # official test split; labels may be absent
        manifest.json
        manifests/{train,val}.json
        foreground/{train,val}/{detections.jsonl,summary.json}  # optional

    Split manifests list samples in ImageFolder order. Original labels are
    retained in sample metadata, including stanford_class_id/name; ImageFolder
    sees one class. The root manifest includes cars_root, annotation paths,
    class_names, and class_count (source classes).
    Official one-based inclusive boxes become original-image, zero-based xyxy
    with exclusive upper bounds. Foreground export can be disabled with
    write_foreground=False or --no-foreground.
    """

    dataset = "Stanford Cars 196"
    foreground_label = "car"

    def __init__(self, source_root: Path):
        super().__init__(source_root)
        self.devkit_root = self.source_root / "devkit"
        self.class_names = _load_class_names(self.devkit_root / "cars_meta.mat")

    def read_records(self) -> list[ImageRecord]:
        records = []
        for split, directory, annotation_name in (
            ("train", "cars_train", "cars_train_annos.mat"),
            ("val", "cars_test", "cars_test_annos.mat"),
        ):
            image_root = self.source_root / directory
            annotation_path = self.devkit_root / annotation_name
            annotations = _load_annotation_structs(annotation_path)
            filenames = [str(item.fname) for item in annotations]
            if len(set(filenames)) != len(filenames):
                raise ValueError(
                    f"Duplicate filenames in Stanford Cars annotations: {annotation_path}"
                )
            for filename in filenames:
                if len(validated_relative_path(filename).parts) != 1:
                    raise ValueError(f"Invalid Stanford Cars filename: {filename}")
            actual = {p.name for p in image_root.iterdir() if p.is_file()}
            if actual != set(filenames):
                raise ValueError(
                    f"Stanford Cars image directory does not match annotations: {image_root}"
                )
            for item, filename in zip(annotations, filenames, strict=True):
                x1, y1, x2, y2 = (
                    _annotation_int(getattr(item, name))
                    for name in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
                )
                field_names = set(getattr(item, "_fieldnames", ()))
                class_id = (
                    _annotation_int(getattr(item, "class"))
                    if "class" in field_names
                    else None
                )
                if split == "train" and class_id is None:
                    raise ValueError(
                        f"Training annotation has no class field: {filename}"
                    )
                if class_id is not None and not 1 <= class_id <= len(self.class_names):
                    raise ValueError(
                        f"Invalid Stanford Cars class ID {class_id} for {filename}"
                    )
                class_name = (
                    self.class_names[class_id - 1] if class_id is not None else None
                )
                records.append(
                    ImageRecord(
                        sample_id=f"{split}/{filename}",
                        source=image_root / filename,
                        split=split,
                        class_name=None,
                        boxes_xyxy=((x1 - 1, y1 - 1, x2, y2),),
                        annotation_source=str(annotation_path),
                        metadata={
                            "source_class_id": class_id,
                            "source_class_name": class_name,
                            "stanford_class_id": class_id,
                            "stanford_class_name": class_name,
                        },
                    )
                )
        return records

    def manifest_metadata(self) -> dict:
        return {
            "cars_root": str(self.source_root),
            "train_annotation_file": str(self.devkit_root / "cars_train_annos.mat"),
            "test_annotation_file": str(self.devkit_root / "cars_test_annos.mat"),
            "class_count": len(self.class_names),
            "class_names": self.class_names,
        }


def stage_cars_dataset(
    cars_root: Path, staged_root: Path, *, write_foreground: bool = True
) -> StageManifest:
    return CarsAdapter(cars_root).stage(staged_root, write_foreground=write_foreground)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage Stanford Cars as ordered train/val ImageFolders with official boxes."
        )
    )
    parser.add_argument("--cars-root", type=Path, required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument(
        "--no-foreground",
        action="store_true",
        help="Skip official foreground cache export",
    )
    args = parser.parse_args()

    manifest = stage_cars_dataset(
        args.cars_root, args.staged_root, write_foreground=not args.no_foreground
    )
    print(
        f"staged Stanford Cars: train={manifest['train_count']} "
        f"val={manifest['val_count']} root={args.staged_root}"
    )


if __name__ == "__main__":
    main()
