"""Small real-image fixtures exercise staging without external datasets/models."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from core.data.cars_stage import stage_cars_dataset
from core.data.common import DatasetAdapter, ImageRecord
from core.data.cub_stage import stage_cub_dataset
from core.data.nabirds_stage import stage_nabirds_dataset
from PIL import Image
from scipy.io import savemat


def image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 10), "red").save(path)


def bird_dataset(root, dataset="cub", boxes=True):
    # Non-numeric class ordering tests ImageFolder order rather than metadata order.
    classes = {2: "002.Bird", 10: "010.Bird"}
    records = [
        ("1", 2, "z.jpg", "1"),
        ("2", 10, "b.jpg", "1"),
        ("3", 2, "a.jpg", "0"),
        ("4", 10, "c.jpg", "0"),
    ]
    values = {
        name: []
        for name in (
            "images",
            "train_test_split",
            "image_class_labels",
            "bounding_boxes",
        )
    }
    for sample_id, class_id, filename, split in records:
        directory = classes[class_id] if dataset == "cub" else str(class_id)
        relative = f"{directory}/{filename}"
        image(root / "images" / relative)
        values["images"].append(f"{sample_id} {relative}\n")
        values["train_test_split"].append(f"{sample_id} {split}\n")
        values["image_class_labels"].append(f"{sample_id} {class_id}\n")
        values["bounding_boxes"].append(f"{sample_id} 1.0 2.0 4.0 5.0\n")
    for name, lines in values.items():
        if name != "bounding_boxes" or boxes:
            (root / f"{name}.txt").write_text("".join(lines))
    (root / "classes.txt").write_text(
        "".join(f"{i} {name}\n" for i, name in classes.items())
    )
    return root


def cars_dataset(root):
    devkit = root / "devkit"
    devkit.mkdir(parents=True)
    savemat(
        devkit / "cars_meta.mat",
        {"class_names": np.array(["Car A", "Car B"], dtype=object)},
    )
    for split, dirname, annofile in (
        ("train", "cars_train", "cars_train_annos.mat"),
        ("val", "cars_test", "cars_test_annos.mat"),
    ):
        annotations = []
        # Unsorted source annotations should still produce ImageFolder order.
        for filename in ("b.jpg", "a.jpg"):
            image(root / dirname / filename)
            record = {
                "fname": filename,
                "bbox_x1": 2,
                "bbox_y1": 3,
                "bbox_x2": 6,
                "bbox_y2": 8,
            }
            if split == "train":
                record["class"] = 2
            annotations.append(record)
        savemat(devkit / annofile, {"annotations": np.array(annotations, dtype=object)})
    return root


@pytest.mark.parametrize("dataset", ["cub", "nabirds", "cars"])
def test_shared_staging_layout_order_boxes_and_idempotence(tmp_path, dataset):
    source = tmp_path / "source"
    output = tmp_path / "staged"
    functions = {
        "cub": stage_cub_dataset,
        "nabirds": stage_nabirds_dataset,
        "cars": stage_cars_dataset,
    }
    if dataset == "cars":
        cars_dataset(source)
    else:
        bird_dataset(source, dataset)
    stage = functions[dataset]
    manifest = stage(source, output)
    assert manifest["train_count"] == manifest["val_count"] == 2
    assert manifest["source_root"] == str(source)
    for split in ("train", "val"):
        payload = json.loads(Path(manifest["split_manifests"][split]).read_text())
        samples = payload["samples"]
        # Compare the actual traversal and class indices used by torchvision.
        from torchvision.datasets import ImageFolder

        actual = ImageFolder(output / split)
        assert payload["class_to_idx"] == actual.class_to_idx
        assert [(s["staged_path"], s["class_index"]) for s in samples] == actual.samples
        digest = hashlib.sha256()
        for path, class_index in actual.samples:
            relative = Path(path).relative_to(output / split).as_posix()
            digest.update(f"{relative}\t{class_index}\n".encode())
        assert payload["source_order_sha256"] == digest.hexdigest()
        foreground = [
            json.loads(line)
            for line in Path(manifest["foreground_caches"][split])
            .read_text()
            .splitlines()
        ]
        assert [r["sample_id"] for r in foreground] == [s["sample_id"] for s in samples]
        assert [r["source_index"] for r in foreground] == [0, 1]
        for record in foreground:
            assert record["width"] == 12 and record["height"] == 10
            assert record["annotation_kind"] == "official"
            assert record["boxes_xyxy"] == (
                [[1, 2, 6, 8]] if dataset == "cars" else [[1, 2, 5, 7]]
            )
        if dataset == "cars":
            assert actual.classes == ["all"]
            assert foreground[0]["stanford_class_id"] == (
                2 if split == "train" else None
            )
        elif dataset == "cub":
            assert actual.classes == ["002.Bird", "010.Bird"]
        else:
            assert actual.classes == ["10", "2"]
        assert all(
            Path(path).is_symlink() and Path(path).is_file()
            for path, _ in actual.samples
        )
    before = (output / "manifest.json").read_bytes()
    assert stage(source, output) == manifest
    assert (output / "manifest.json").read_bytes() == before


@pytest.mark.parametrize(
    "failure",
    [
        "duplicate_id",
        "split_flag",
        "missing_id",
        "missing_image",
        "invalid_box",
        "path_escape",
    ],
)
def test_bird_validation_finishes_before_writing(tmp_path, failure):
    source = bird_dataset(tmp_path / "source")
    if failure == "duplicate_id":
        with (source / "images.txt").open("a") as stream:
            stream.write("1 002.Bird/z.jpg\n")
    elif failure == "split_flag":
        p = source / "train_test_split.txt"
        p.write_text(p.read_text().replace("1 1", "1 9"))
    elif failure == "missing_id":
        (source / "image_class_labels.txt").write_text("1 2\n")
    elif failure == "missing_image":
        (source / "images/002.Bird/z.jpg").unlink()
    elif failure == "invalid_box":
        p = source / "bounding_boxes.txt"
        p.write_text(p.read_text().replace("4.0 5.0", "400.0 5.0"))
    else:
        p = source / "images.txt"
        p.write_text(p.read_text().replace("002.Bird/z.jpg", "../outside.jpg"))
    output = tmp_path / "staged"
    with pytest.raises((ValueError, FileNotFoundError)):
        stage_cub_dataset(source, output)
    assert not output.exists()


def test_nabirds_relative_roots_and_rejects_stale_files(tmp_path, monkeypatch):
    bird_dataset(tmp_path / "source", "nabirds")
    monkeypatch.chdir(tmp_path)
    manifest = stage_nabirds_dataset(Path("source"), Path("staged"))
    link = tmp_path / "staged/train/2/z.jpg"
    assert link.is_file()
    assert link.readlink().is_absolute()
    stale = tmp_path / "staged/train/2/old.jpg"
    image(stale)
    before = (tmp_path / "staged/manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="Unexpected"):
        stage_nabirds_dataset(Path("source"), Path("staged"))
    assert stale.is_file()
    assert (tmp_path / "staged/manifest.json").read_bytes() == before
    assert manifest["train_count"] == 2


@pytest.mark.parametrize(
    "obstruction",
    ["real_file", "class_symlink", "split_symlink", "duplicate_destination"],
)
def test_output_validation_does_not_overwrite_files(tmp_path, obstruction):
    source = bird_dataset(tmp_path / "source", "nabirds")
    output = tmp_path / "staged"
    if obstruction == "real_file":
        image(output / "train/2/z.jpg")
    elif obstruction in {"class_symlink", "split_symlink"}:
        target = tmp_path / "unrelated"
        target.mkdir()
        destination = output / (
            "train/2" if obstruction == "class_symlink" else "train"
        )
        destination.parent.mkdir(parents=True)
        destination.symlink_to(target, target_is_directory=True)
    else:
        p = source / "images.txt"
        p.write_text(p.read_text() + "5 2/z.jpg\n")
        for filename, line in (
            ("train_test_split.txt", "5 1\n"),
            ("image_class_labels.txt", "5 2\n"),
            ("bounding_boxes.txt", "5 1 2 4 5\n"),
        ):
            with (source / filename).open("a") as stream:
                stream.write(line)
    with pytest.raises((FileExistsError, ValueError)):
        stage_nabirds_dataset(source, output)
    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize("boxes,export", [(False, True), (True, False)])
def test_staging_without_foreground_cache(tmp_path, boxes, export):
    source = bird_dataset(tmp_path / "source", boxes=boxes)
    output = tmp_path / "staged"
    manifest = stage_cub_dataset(source, output, write_foreground=export)
    assert manifest["foreground_caches"] == {}
    assert not (output / "foreground").exists()
    assert (output / "manifests/train.json").exists()


def test_new_adapter_only_needs_records(tmp_path):
    class NewDataset(DatasetAdapter):
        dataset = "example"

        def read_records(self):
            return [
                ImageRecord("1", self.source_root / "one.jpg", "train"),
                ImageRecord("2", self.source_root / "two.jpg", "val"),
            ]

    source = tmp_path / "source"
    image(source / "one.jpg")
    image(source / "two.jpg")
    manifest = NewDataset(source).stage(tmp_path / "staged")
    assert manifest["staged_class_counts"] == {"train": 1, "val": 1}
    assert manifest["foreground_caches"] == {}


def test_missing_annotation_is_distinct_from_no_detection(tmp_path):
    class PartialBoxes(DatasetAdapter):
        dataset = "partial-boxes"

        def read_records(self):
            return [
                ImageRecord(
                    "1",
                    self.source_root / "a.jpg",
                    "train",
                    boxes_xyxy=((0, 0, 4, 5),),
                    annotation_source="official.txt",
                ),
                ImageRecord("2", self.source_root / "b.jpg", "train"),
                ImageRecord("3", self.source_root / "c.jpg", "val"),
            ]

    source = tmp_path / "source"
    for filename in ("a.jpg", "b.jpg", "c.jpg"):
        image(source / filename)
    manifest = PartialBoxes(source).stage(tmp_path / "staged")
    assert set(manifest["foreground_caches"]) == {"train"}
    path = Path(manifest["foreground_caches"]["train"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["status"] for row in rows] == ["detected", "missing_annotation"]
    summary = json.loads((path.parent / "summary.json").read_text())
    assert summary["missing_annotation_images"] == 1
    assert summary["no_detection_images"] == 0
    assert summary["detected_images"] == 1


def test_annotated_filename_is_preserved_for_symlink_sources(tmp_path):
    source = bird_dataset(tmp_path / "source", "nabirds")
    original = source / "images/2/z.jpg"
    target = source / "other-name.jpg"
    original.rename(target)
    original.symlink_to(target)
    output = tmp_path / "staged"
    manifest = stage_nabirds_dataset(source, output)
    assert (output / "train/2/z.jpg").resolve() == target
    payload = json.loads(Path(manifest["split_manifests"]["train"]).read_text())
    assert any(s["relative_path"] == "2/z.jpg" for s in payload["samples"])


def test_cub_cli_stages_images(tmp_path, monkeypatch):
    from core.data.cub_stage import main

    source = bird_dataset(tmp_path / "source")
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        ["cub_stage", "--cub-root", str(source), "--staged-root", str(output)],
    )
    main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["train_count"] == 2
    assert (output / "train/002.Bird/z.jpg").is_file()
