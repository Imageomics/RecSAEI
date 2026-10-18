"""Read official NABirds metadata for the shared staging pipeline."""

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
from .common import (
    read_hierarchy as read_hierarchy,
)  # Preserve the previous public reader.


class NABirdsAdapter(DatasetAdapter):
    """Stage NABirds with official class IDs as ImageFolder directories.

    Output under staged_root::

        train/<class_id>/<filename>  # official training split
        val/<class_id>/<filename>    # official test split
        manifest.json
        manifests/{train,val}.json
        foreground/{train,val}/{detections.jsonl,summary.json}  # optional

    Split manifests list samples in ImageFolder order with source class IDs and
    names in sample metadata. The root manifest includes nabirds_root,
    split_file, and class_count (image-bearing source classes).
    Optional official xywh boxes become original-image, zero-based xyxy with
    exclusive upper bounds. Foreground export can be disabled with
    write_foreground=False or --no-foreground.
    """

    dataset = "NABirds"
    foreground_label = "bird"

    def read_records(self) -> list[ImageRecord]:
        root = self.source_root
        images = read_id_map(root / "images.txt")
        flags = read_id_map(root / "train_test_split.txt")
        labels = read_id_map(root / "image_class_labels.txt")
        classes = read_classes(root / "classes.txt")
        validate_matching_ids(images, split_flags=flags, class_ids=labels)
        if set(flags.values()) - {"0", "1"}:
            raise ValueError("NABirds split flags must be 0 or 1")
        box_path = root / "bounding_boxes.txt"
        boxes = read_bounding_boxes(box_path) if box_path.exists() else {}
        if box_path.exists():
            validate_matching_ids(images, bounding_boxes=boxes)
        records = []
        for image_id, text in images.items():
            relpath = validated_relative_path(text)
            class_id = int(labels[image_id])
            if class_id not in classes:
                raise ValueError(f"Unknown NABirds class ID {class_id} for {image_id}")
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
                    split="train" if flags[image_id] == "1" else "val",
                    class_name=labels[image_id],
                    boxes_xyxy=xyxy,
                    annotation_source=str(box_path) if box is not None else None,
                    metadata={
                        "source_class_id": class_id,
                        "source_class_name": classes[class_id],
                    },
                )
            )
        return records

    def manifest_metadata(self) -> dict:
        root = self.source_root
        labels = read_id_map(root / "image_class_labels.txt")
        return {
            "nabirds_root": str(root),
            "class_count": len(set(labels.values())),
            "split_file": str(root / "train_test_split.txt"),
        }


def stage_nabirds_dataset(
    nabirds_root: Path, staged_root: Path, *, write_foreground: bool = True
) -> StageManifest:
    return NABirdsAdapter(nabirds_root).stage(
        staged_root, write_foreground=write_foreground
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create train/val ImageFolder symlink trees for NABirds."
    )
    parser.add_argument("--nabirds-root", type=Path, required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument(
        "--no-foreground",
        action="store_true",
        help="Skip official foreground cache export",
    )
    args = parser.parse_args()

    manifest = stage_nabirds_dataset(
        args.nabirds_root, args.staged_root, write_foreground=not args.no_foreground
    )
    print(
        f"staged NABirds: train={manifest['train_count']} "
        f"val={manifest['val_count']} root={args.staged_root}"
    )


if __name__ == "__main__":
    main()
