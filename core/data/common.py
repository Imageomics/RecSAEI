"""Dataset adapter contract and metadata readers shared by staging tools."""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, TypedDict

import numpy as np
from core.tools.io import dump_json
from PIL import Image
from tqdm import tqdm

Box = tuple[float, float, float, float]

BOX_CONVENTION = "original-image pixels; zero-based xyxy; exclusive upper bounds"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class ImageRecord:
    """One source image; boxes use original-image, zero-based, exclusive xyxy.

    ``class_name`` controls the staged directory; None means ``all``.
    Source labels belong in ``metadata`` when they do not define directories.
    """

    sample_id: str
    source: Path
    split: str
    class_name: str | None = None
    boxes_xyxy: tuple[Box, ...] = ()
    annotation_source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StageManifest(TypedDict):
    schema_version: int
    dataset: str
    source_root: str
    staged_root: str
    test_split_name: str
    train_root: str
    val_root: str
    train_count: int
    val_count: int
    staged_class_counts: dict[str, int]
    split_manifests: dict[str, str]
    foreground_caches: dict[str, str]


class DatasetAdapter(ABC):
    """Read a dataset and save its staged images and corresponding artifacts.

    Subclasses set ``dataset`` and implement ``read_records()`` without writing
    files. Records must cover non-empty train/val splits with unique sample IDs
    within each split. Optional boxes require an annotation source.
    ``manifest_metadata()`` can add dataset-specific fields to the manifest.

    Example::

        class ExampleAdapter(DatasetAdapter):
            dataset = "Example"

            def read_records(self) -> list[ImageRecord]:
                return [
                    ImageRecord("1", self.source_root / "a.jpg", "train"),
                    ImageRecord("2", self.source_root / "b.jpg", "val"),
                ]

        ExampleAdapter(Path("/data/example")).stage(Path("/data/staged"))

    Call ``stage()`` to read, validate, and save the dataset. See each adapter
    for its folder layout and output metadata.
    """

    dataset: str
    foreground_label: str = "object"

    def __init__(self, source_root: Path):
        self.source_root = source_root.resolve()

    @abstractmethod
    def read_records(self) -> list[ImageRecord]:
        """Translate source metadata to image records without writing files."""

    def manifest_metadata(self) -> dict[str, Any]:
        return {}

    def stage(
        self, staged_root: Path, *, write_foreground: bool = True
    ) -> StageManifest:
        """Validate everything, then stage links, ordered manifests, and official boxes.

        Rerunning identical inputs is supported. Unexpected existing split entries
        are rejected rather than deleted. Model-based detection is a separate step.
        """
        staged_root = staged_root.resolve()
        if staged_root == self.source_root or self.source_root.is_relative_to(
            staged_root
        ):
            raise ValueError("Staging output must not contain the source dataset")
        records = self._validate_records(self.read_records(), staged_root)
        by_split = {
            split: [r for r in records if r.split == split]
            for split in ("train", "val")
        }
        if any(not items for items in by_split.values()):
            raise ValueError("Both train and val splits must contain images")
        # Build and validate all manifests (including boxes) before creating links.
        payloads = {
            split: self._split_manifest(items, split, staged_root)
            for split, items in by_split.items()
        }
        extras = self.manifest_metadata()
        manifest: StageManifest = {
            "schema_version": 1,
            "dataset": self.dataset,
            "source_root": str(self.source_root),
            "staged_root": str(staged_root),
            "test_split_name": "val",
            "train_root": str(staged_root / "train"),
            "val_root": str(staged_root / "val"),
            "train_count": len(by_split["train"]),
            "val_count": len(by_split["val"]),
            "staged_class_counts": {s: p["n_classes"] for s, p in payloads.items()},
            "split_manifests": {},
            "foreground_caches": {},
        }
        if set(extras) & set(manifest):
            raise ValueError(
                "Adapter manifest metadata must not override shared fields"
            )
        manifest.update(extras)
        # Catch unsupported adapter metadata before creating any output.
        json.dumps(manifest)
        for payload in payloads.values():
            json.dumps(payload)
        for record in tqdm(
            records,
            desc=f"stage-{self.dataset}",
            unit="image",
            disable=len(records) < 100,
        ):
            replace_symlink(record.source, staged_root / _relative_path(record))
        (staged_root / "manifests").mkdir(parents=True, exist_ok=True)
        for split, payload in payloads.items():
            manifest_path = staged_root / "manifests" / f"{split}.json"
            dump_json(payload, manifest_path)
            manifest["split_manifests"][split] = str(manifest_path)
            if write_foreground and any(r.boxes_xyxy for r in by_split[split]):
                path = self._write_foreground_annotations(
                    payload,
                    manifest_path,
                    staged_root / "foreground" / split,
                )
                manifest["foreground_caches"][split] = str(path)
        dump_json(manifest, staged_root / "manifest.json")
        return manifest

    def _validate_records(
        self, records: list[ImageRecord], staged_root: Path
    ) -> list[ImageRecord]:
        if not records:
            raise ValueError("Dataset contains no image records")
        validated = []
        ids = set()
        expected = set()
        for record in records:
            if record.split not in {"train", "val"}:
                raise ValueError(
                    f"Unsupported split {record.split!r}; expected train or val"
                )
            if record.class_name is not None:
                _component(record.class_name)
            if not record.sample_id or (record.split, record.sample_id) in ids:
                raise ValueError(
                    f"Empty or duplicate sample ID: {record.split}/{record.sample_id}"
                )
            ids.add((record.split, record.sample_id))
            source = record.source.resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Missing source image: {source}")
            if source.is_relative_to(staged_root):
                raise ValueError(f"Source image is inside the staging output: {source}")
            if source.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported ImageFolder image extension: {source}")
            # Preserve the annotated filename even when the source itself is a symlink.
            normalized = replace(record, source=record.source.absolute())
            relative = _relative_path(normalized)
            if relative in expected:
                raise ValueError(f"Duplicate staging destination: {relative}")
            expected.add(relative)
            validated.append(normalized)

        # Inspect all existing entries before making any output changes. Extra files
        # or class directories would change ImageFolder order and invalidate indices.
        expected_dirs = {p.parent for p in expected} | {
            Path(p.parts[0]) for p in expected
        }
        if staged_root.exists() and not staged_root.is_dir():
            raise FileExistsError(f"Staging root is not a directory: {staged_root}")
        for split in ("train", "val"):
            split_root = staged_root / split
            if split_root.is_symlink():
                raise FileExistsError(
                    f"Staged split must not be a symlink: {split_root}"
                )
            if not split_root.exists():
                continue
            if not split_root.is_dir():
                raise FileExistsError(f"Staged split is not a directory: {split_root}")
            for path in split_root.rglob("*"):
                relative = path.relative_to(staged_root)
                if relative in expected:
                    if not path.is_symlink():
                        raise FileExistsError(
                            f"Refusing to overwrite non-symlink path: {path}"
                        )
                elif (
                    relative in expected_dirs
                    and path.is_dir()
                    and not path.is_symlink()
                ):
                    continue
                else:
                    raise FileExistsError(f"Unexpected existing staged entry: {path}")
        return sorted(validated, key=lambda r: _relative_path(r).as_posix())

    def _split_manifest(
        self, records: list[ImageRecord], split: str, staged_root: Path
    ) -> dict[str, Any]:
        classes = sorted({r.class_name or "all" for r in records})
        class_to_idx = {name: i for i, name in enumerate(classes)}
        class_counts = Counter(r.class_name or "all" for r in records)
        digest = hashlib.sha256()
        samples = []
        for i, record in enumerate(records):
            name = record.class_name or "all"
            relative = Path(name) / record.source.name
            class_index = class_to_idx[name]
            digest.update(f"{relative.as_posix()}\t{class_index}\n".encode())
            # Reading image headers also validates dimensions before output starts.
            with Image.open(record.source) as image:
                width, height = image.size
            for box in record.boxes_xyxy:
                if len(box) != 4 or not all(math.isfinite(v) for v in box):
                    raise ValueError(f"Invalid bounding box for {record.source}: {box}")
                x1, y1, x2, y2 = box
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(
                        f"Bounding box {box} is outside {width}x{height}: {record.source}"
                    )
            if record.boxes_xyxy and not record.annotation_source:
                raise ValueError(f"Boxes require an annotation source: {record.source}")
            samples.append(
                {
                    "sample_id": record.sample_id,
                    "source_index": i,
                    "source_path": str(record.source.resolve()),
                    "staged_path": str(staged_root / split / relative),
                    "relative_path": relative.as_posix(),
                    "class_index": class_index,
                    "class_name": name,
                    "width": width,
                    "height": height,
                    "boxes_xyxy": [list(box) for box in record.boxes_xyxy],
                    "annotation_source": record.annotation_source,
                    "metadata": record.metadata,
                }
            )
        return {
            "schema_version": 1,
            "dataset_root": str(staged_root / split),
            "source_split": "test" if split == "val" else split,
            "n_classes": len(classes),
            "n_images": len(records),
            "source_order_sha256": digest.hexdigest(),
            "assignment_sha256": hashlib.sha256(
                np.arange(len(records), dtype=np.int64).tobytes()
            ).hexdigest(),
            "splits": {
                "all": {
                    "n_images": len(records),
                    "source_indices": list(range(len(records))),
                }
            },
            "classes": [
                {
                    "class_index": class_to_idx[name],
                    "class_name": name,
                    "total": class_counts[name],
                }
                for name in classes
            ],
            "class_to_idx": class_to_idx,
            "box_convention": BOX_CONVENTION,
            "samples": samples,
        }

    def _write_foreground_annotations(
        self, split_manifest: dict[str, Any], manifest_path: Path, output_dir: Path
    ) -> Path:
        """Export boxes in the exact sample order recorded by the split manifest."""
        label = self.foreground_label
        output_dir.mkdir(parents=True, exist_ok=True)
        detections_path = output_dir / "detections.jsonl"
        per_class = {}
        detected = missing = multiple = 0
        with detections_path.open("w", encoding="utf-8") as stream:
            for sample in split_manifest["samples"]:
                boxes = sample["boxes_xyxy"]
                status = "detected" if boxes else "missing_annotation"
                detected += bool(boxes)
                missing += not boxes
                multiple += len(boxes) > 1
                row = {
                    **sample["metadata"],
                    "sample_id": sample["sample_id"],
                    "source_index": sample["source_index"],
                    "source_path": sample["source_path"],
                    "staged_path": sample["staged_path"],
                    "class_index": sample["class_index"],
                    "synset": sample["class_name"],
                    "class_name": label,
                    "prompt": label,
                    "width": sample["width"],
                    "height": sample["height"],
                    "status": status,
                    "boxes_xyxy": boxes,
                    "box_convention": BOX_CONVENTION,
                    "annotation_kind": "official" if boxes else None,
                    "annotation_source": sample["annotation_source"],
                    "scores": [1.0] * len(boxes),
                    "labels": ["official bounding box"] * len(boxes),
                }
                stream.write(json.dumps(row) + "\n")
                counts = per_class.setdefault(
                    sample["class_index"],
                    {
                        "class_index": sample["class_index"],
                        "synset": sample["class_name"],
                        "class_name": label,
                        "completed": 0,
                        "detected": 0,
                        "no_detection": 0,
                        "missing_annotation": 0,
                    },
                )
                counts["completed"] += 1
                counts["detected"] += bool(boxes)
                counts["missing_annotation"] += not boxes
        for counts in per_class.values():
            counts["detection_rate"] = counts["detected"] / counts["completed"]
        samples = split_manifest["samples"]
        summary = {
            "schema_version": 1,
            "description": "Official dataset foreground bounding boxes.",
            "dataset_root": split_manifest["dataset_root"],
            "split_manifest_path": str(manifest_path),
            "split": "all",
            "source_order_sha256": split_manifest["source_order_sha256"],
            "assignment_sha256": split_manifest["assignment_sha256"],
            "annotation_sources": sorted(
                {s["annotation_source"] for s in samples if s["annotation_source"]}
            ),
            "box_convention": BOX_CONVENTION,
            "requested_images": len(samples),
            "completed_images": len(samples),
            "pending_images": 0,
            "detected_images": detected,
            "no_detection_images": 0,
            "missing_annotation_images": missing,
            "multiple_detection_images": multiple,
            "detection_rate": detected / len(samples) if samples else 0.0,
            "detections_path": str(detections_path),
            "per_class": [per_class[i] for i in sorted(per_class)],
        }
        dump_json(summary, output_dir / "summary.json")
        return detections_path


def read_id_map(path: Path) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected an ID and value")
        key, value = fields
        if key in pairs:
            raise ValueError(f"{path}:{line_number}: duplicate ID {key}")
        pairs[key] = value
    return pairs


def read_classes(path: Path) -> dict[int, str]:
    classes: dict[int, str] = {}
    for key, value in read_id_map(path).items():
        class_id = int(key)
        if class_id in classes:
            raise ValueError(f"{path}: duplicate numeric class ID {class_id}")
        classes[class_id] = value
    return classes


def validate_matching_ids(
    reference: Mapping[str, Any], **tables: Mapping[str, Any]
) -> None:
    expected = set(reference)
    for name, table in tables.items():
        if set(table) != expected:
            missing = sorted(expected - set(table))[:5]
            extra = sorted(set(table) - expected)[:5]
            raise ValueError(
                f"{name} IDs do not match images: missing={missing}, extra={extra}"
            )


def validated_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Invalid relative image path: {value}")
    return path


def read_bounding_boxes(path: Path) -> dict[str, tuple[int, int, int, int]]:
    """Read NABirds/CUB integer-valued xywh metadata (including '1.0')."""
    boxes = {}
    for image_id, text in read_id_map(path).items():
        values = [float(value) for value in text.split()]
        if len(values) != 4 or not all(
            math.isfinite(v) and v.is_integer() for v in values
        ):
            raise ValueError(
                f"{path}: invalid integer bounding box for {image_id}: {text}"
            )
        x, y, width, height = (int(value) for value in values)
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError(
                f"{path}: invalid xywh bounding box for {image_id}: {text}"
            )
        boxes[image_id] = (x, y, width, height)
    return boxes


def read_hierarchy(path: Path) -> dict[int, int]:
    return {int(child): int(parent) for child, parent in read_id_map(path).items()}


def replace_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(f"Refusing to overwrite non-symlink path: {destination}")
    destination.symlink_to(source.resolve())


def _component(value: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"Invalid staged path component: {value!r}")
    return value


def _relative_path(record: ImageRecord) -> Path:
    return Path(record.split) / (record.class_name or "all") / record.source.name
