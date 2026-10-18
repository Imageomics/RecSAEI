"""Read official CUB metadata for the shared staging pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import (
    DatasetAdapter,
    ImageRecord,
    StageManifest,
    read_bounding_boxes,
    read_classes,
    read_id_map,
    validate_matching_ids,
    validated_relative_path,
)


class CUBAdapter(DatasetAdapter):
    """Stage CUB with official class names as ImageFolder directories.

    Output under staged_root::

        train/<class_name>/<filename>  # official training split
        val/<class_name>/<filename>    # official test split
        manifest.json
        manifests/{train,val}.json
        foreground/{train,val}/{detections.jsonl,summary.json}  # optional

    Split manifests list samples in ImageFolder order with source class IDs and
    names in sample metadata. The root manifest includes cub_root, split_file,
    and class_count. Every source class must occur in both splits.
    Optional official xywh boxes become original-image, zero-based xyxy with
    exclusive upper bounds. Foreground export can be disabled with
    write_foreground=False or --no-foreground.

    """

    dataset = "CUB_200_2011"
    foreground_label = "bird"

    def read_records(self) -> list[ImageRecord]:
        root = self.source_root
        images = read_id_map(root / "images.txt")
        flags = read_id_map(root / "train_test_split.txt")
        labels = read_id_map(root / "image_class_labels.txt")
        classes = read_classes(root / "classes.txt")
        validate_matching_ids(images, split_flags=flags, class_ids=labels)
        if set(flags.values()) - {"0", "1"}:
            raise ValueError("CUB split flags must be 0 or 1")
        box_path = root / "bounding_boxes.txt"
        boxes = read_bounding_boxes(box_path) if box_path.exists() else {}
        if box_path.exists():
            validate_matching_ids(images, bounding_boxes=boxes)
        records = []
        classes_by_split = {"train": set(), "val": set()}
        for image_id, text in images.items():
            relpath = validated_relative_path(text)
            class_id = int(labels[image_id])
            if class_id not in classes:
                raise ValueError(f"Unknown CUB class ID {class_id} for {image_id}")
            class_name = classes[class_id]
            if len(relpath.parts) != 2 or relpath.parent.name != class_name:
                raise ValueError(
                    f"CUB image {image_id} path does not match class {class_name}: {relpath}"
                )
            split = "train" if flags[image_id] == "1" else "val"
            classes_by_split[split].add(class_id)
            box = boxes.get(image_id)
            xyxy = (
                ()
                if box is None
                else ((box[0], box[1], box[0] + box[2], box[1] + box[3]),)
            )
            records.append(
                ImageRecord(
                    sample_id=image_id,
                    source=root / "images" / relpath,
                    split=split,
                    class_name=class_name,
                    boxes_xyxy=xyxy,
                    annotation_source=str(box_path) if box is not None else None,
                    metadata={
                        "source_class_id": class_id,
                        "source_class_name": class_name,
                    },
                )
            )
        for split, present in classes_by_split.items():
            if present != set(classes):
                raise ValueError(
                    f"CUB {split} split is missing classes: {sorted(set(classes) - present)}"
                )
        return records

    def manifest_metadata(self) -> dict:
        root = self.source_root
        return {
            "cub_root": str(root),
            "class_count": len(read_classes(root / "classes.txt")),
            "split_file": str(root / "train_test_split.txt"),
        }


def stage_cub_dataset(
    cub_root: Path, staged_root: Path, *, write_foreground: bool = True
) -> StageManifest:
    return CUBAdapter(cub_root).stage(staged_root, write_foreground=write_foreground)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create train/val ImageFolder symlink trees from the official "
            "CUB-200-2011 train/test split."
        )
    )
    parser.add_argument("--cub-root", type=Path, required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument(
        "--no-foreground",
        action="store_true",
        help="Skip official foreground cache export",
    )
    args = parser.parse_args()

    manifest = stage_cub_dataset(
        args.cub_root, args.staged_root, write_foreground=not args.no_foreground
    )
    print(
        f"staged CUB: train={manifest['train_count']} "
        f"val={manifest['val_count']} root={args.staged_root}"
    )


if __name__ == "__main__":
    main()
