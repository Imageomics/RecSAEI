"""Selection, export layout, and browser API behavior without pretrained models."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from core.selection import manual_dimensions
from core.selection.activation import select_activation
from core.selection.class_specific import fit_class_scores, select_class
from fastapi.testclient import TestClient
from PIL import Image
from scipy import sparse
from visualization.api import create_app
from visualization.export import export_viewer
from visualization.rendering import display_image, overlay


def source(tmp_path, name, mode, *, dataset=None):
    dataset = dataset or tmp_path / "images"
    for label, color in [("A", "red"), ("B", "blue")]:
        for i in range(2):
            path = dataset / label / f"{i}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (12, 6), color).save(path)
    folder = tmp_path / name
    shards = folder / "shards"
    inference = folder / "inference"
    checkpoint = folder / "run/checkpoint"
    for path in [shards, inference, checkpoint]:
        path.mkdir(parents=True)
    metadata = {
        "family": "dinov3",
        "ckpt": "dinov3.pth",
        "encoder_implementation": "official",
        "image_resize_mode": "square",
        "content_tokens_per_example": 4,
        "cls_token": True,
        "d_model": 3,
        "n_examples": 4,
        "dataset": str(dataset),
        "layers": [-2, -1],
    }
    (shards / "metadata.json").write_text(json.dumps(metadata))
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "train_data": {
                    "tokens": "special" if mode == "cls" else "content",
                    "layer": -1 if mode == "cls" else -2,
                }
            }
        )
    )
    pooled = np.array(
        [[8.0, 1.0, 0.0], [5.0, 2.0, 0.0], [0.0, 9.0, 1.0], [0.0, 6.0, 2.0]],
        dtype=np.float32,
    )
    codes = pooled if mode == "cls" else np.repeat(pooled, 4, axis=0)
    sparse.save_npz(inference / "token_acts.npz", sparse.csr_array(codes))
    return {
        "run_dir": str(folder / "run"),
        "shards_dir": str(shards),
        "inference_dir": str(inference),
    }


@pytest.fixture
def exported(tmp_path):
    patch = source(tmp_path, "patch", "patch")
    cls = source(tmp_path, "cls", "cls")
    patch["fit"] = dict(patch)
    root = tmp_path / "export"
    config = {
        "title": "Test viewer",
        "output_dir": str(root),
        "sources": {"patch": patch, "cls": cls},
    }
    export_viewer(config)
    return root, config


def test_selection_pooling_and_region():
    codes = sparse.csr_array([[9.0, 2.0, 1.0], [0.0, 8.0, 1.0]])
    assert select_activation(codes, pooling="mean")[0]["dim"] == 1
    assert select_activation(codes, pooling="max")[0]["dim"] == 0
    assert select_activation(codes, patches=[0])[0]["dim"] == 0
    with pytest.raises(ValueError, match="patch"):
        select_activation(codes, patches=[2])
    assert [item["dim"] for item in manual_dimensions([2, 0], d_sae=3)] == [2, 0]
    with pytest.raises(ValueError, match="unique"):
        manual_dimensions([1, 1], d_sae=3)


def test_class_separation():
    scores = fit_class_scores(
        [[8, 0, 0], [6, 0, 0], [0, 9, 0], [0, 7, 0]], ["A", "A", "B", "B"]
    )
    assert [row["dim"] for row in select_class(scores, "A")] == [0]
    assert select_class(scores, "B")[0]["dim"] == 1


def test_export_distinguishes_patch_and_cls(exported):
    root, _ = exported
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["modes"]["patch"]["tokens_per_image"] == 4
    assert manifest["modes"]["cls"]["tokens_per_image"] == 1
    assert sparse.load_npz(root / "patch/token_acts.npz").shape == (16, 3)
    assert sparse.load_npz(root / "cls/token_acts.npz").shape == (4, 3)
    assert manifest["modes"]["patch"]["class_selection"]


def test_export_rejects_wrong_layout(exported):
    _, config = exported
    path = Path(config["sources"]["cls"]["inference_dir"]) / "token_acts.npz"
    sparse.save_npz(path, sparse.csr_array(np.ones((16, 3))))
    with pytest.raises(ValueError, match="layout"):
        export_viewer(config)


def test_offline_api_rank_values_and_gallery(exported):
    root, _ = exported
    with TestClient(create_app(root)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/_viewer/app.js").status_code == 200
        assert client.get("/api/manifest").json()["live_enabled"] is False
        assert "source_path" not in client.get("/api/images").json()["items"][0]
        response = client.get(
            "/api/dimensions",
            params={"mode": "patch", "method": "class_specific", "class_name": "A"},
        )
        assert response.status_code == 200
        assert response.json()["items"][0]["dim"] == 0
        response = client.get(
            "/api/dimensions",
            params=[("mode", "cls"), ("method", "manual"), ("dim", 2), ("dim", 0)],
        )
        assert [item["dim"] for item in response.json()["items"]] == [2, 0]
        assert client.get(
            "/api/images/0/values", params={"mode": "cls", "dim": 0}
        ).json()["values"] == [8.0]
        assert (
            len(
                client.get(
                    "/api/images/0/values", params={"mode": "patch", "dim": 0}
                ).json()["values"]
            )
            == 4
        )
        assert client.get("/api/images/0/overlay?mode=cls&dim=0").status_code == 400
        assert (
            client.get("/api/images/0/overlay?mode=patch&dim=0").headers["content-type"]
            == "image/png"
        )
        gallery = client.get("/api/dimensions/1?mode=patch").json()["items"]
        assert gallery[0]["image_id"] == "2"
        assert client.get("/api/dimensions/99?mode=patch").status_code == 400
        assert client.get("/api/images/nope/image?mode=patch").status_code == 404
        assert client.get("/api/images/0/reconstruction").status_code == 400
        assert (
            client.get("/api/dimensions?mode=cls&image_id=0&patch=0").status_code == 400
        )


def test_live_api_dispatch_and_checkpoint_identity(exported):
    root, config = exported
    calls = []

    class FakeRuntime:
        def __init__(self, config):
            calls.append("load")

        def edit(self, path, request):
            calls.append(request.mode)
            return SimpleNamespace(
                image=torch.zeros(3, 2, 2), details={"mode": request.mode}
            )

        def reconstruct(self, path):
            return torch.zeros(3, 2, 2)

        def patch_attribution(self, path, *, target_dimension):
            calls.append(("patch_attribution", target_dimension))
            return [1.0, 0, 0, 0]

    runtime = {
        f"{mode}_checkpoint": str(Path(source["run_dir"]) / "checkpoint/sae.pt")
        for mode, source in config["sources"].items()
    }
    with TestClient(
        create_app(root, runtime_config=runtime, runtime_factory=FakeRuntime)
    ) as client:
        assert calls == ["load"]
        for mode in ["patch", "cls"]:
            response = client.post(
                "/api/images/0/edit",
                json={"mode": mode, "dimensions": [0], "strength": 0},
            )
            assert response.status_code == 200
            assert response.json()["image"].startswith("data:image/png;base64,")
        assert calls == ["load", "patch", "cls"]
        assert client.post(
            "/api/images/0/attribution", json={"target_dimension": 0}
        ).status_code == 404
        assert client.get("/api/images/0/reconstruction").status_code == 200
        response = client.get("/api/images/0/overlay?mode=cls&dim=0")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert calls[-1] == ("patch_attribution", 0)
        assert (
            client.post(
                "/api/images/0/edit",
                json={
                    "mode": "patch",
                    "dimensions": [0],
                    "strength": 1,
                    "patches": [100],
                },
            ).status_code
            == 400
        )
    runtime["patch_checkpoint"] = "/wrong/sae.pt"
    with pytest.raises(ValueError, match="checkpoint differs"):
        with TestClient(
            create_app(root, runtime_config=runtime, runtime_factory=FakeRuntime)
        ):
            pass
    assert calls == ["load", "patch", "cls", ("patch_attribution", 0)]


def test_spatial_overlay_coordinates(tmp_path):
    path = tmp_path / "wide.png"
    Image.new("RGB", (20, 10), "white").save(path)
    image = display_image(path, 32)
    assert image.size == (32, 32)
    result = np.asarray(overlay(image, [1, 0, 0, 0]))
    assert result[0, 0, 1] < 255
    np.testing.assert_array_equal(result[20, 20], [255, 255, 255])


def test_runtime_uses_current_interventions(tmp_path, monkeypatch):
    from core.intervention import RAEv2Backend
    from test_intervention import ToyRAE, ToySAE
    from visualization.api import EditInput
    from visualization.runtime import LiveRuntime

    backend = RAEv2Backend(
        rae=ToyRAE(), stage1_params={"encoder_name": "dinov3-vit-l16", "resolution": 2}
    )
    monkeypatch.setattr(RAEv2Backend, "from_config", lambda *args, **kwargs: backend)
    monkeypatch.setattr(backend, "load_sae", lambda *args, **kwargs: ToySAE())
    image = tmp_path / "image.png"
    pixels = np.array(
        [[[30, 80, 120], [40, 90, 150]], [[60, 110, 160], [70, 130, 170]]],
        dtype=np.uint8,
    )
    Image.fromarray(pixels).save(image)
    runtime = LiveRuntime(
        {
            "stage1_config": "unused",
            "device": "cpu",
            "patch_checkpoint": "patch.pt",
            "cls_checkpoint": "cls.pt",
            "cache_size": 1,
        }
    )
    original = runtime.reconstruct(str(image))
    for mode, identity in [("patch", 1.0), ("cls", 0.0)]:
        request = EditInput(
            mode=mode, dimensions=[0], strength=identity, source_dimensions=3
        )
        result = runtime.edit(str(image), request)
        torch.testing.assert_close(result.image, original, atol=0, rtol=0)
    scores = runtime.patch_attribution(str(image), target_dimension=0)
    assert len(scores) == 4
    assert np.isfinite(scores).all() and np.all(np.asarray(scores) > 0)
    # Spatial attribution needs only the CLS SAE, not a patch SAE.
    runtime.patch_sae = None
    np.testing.assert_allclose(
        runtime.patch_attribution(str(image), target_dimension=0),
        scores,
    )
    assert len(runtime.cache) == 1


def test_export_resolves_current_saev_inference_directory(tmp_path):
    from visualization.export import load_source

    config = source(tmp_path, "original", "patch")
    original = Path(config["shards_dir"])
    metadata = json.loads((original / "metadata.json").read_text())
    metadata.update(
        max_tokens_per_shard=100, data="", pixel_agg="majority", tag="viewer-test"
    )
    shards = tmp_path / "saev/shards/viewer-test"
    shards.mkdir(parents=True)
    (shards / "metadata.json").write_text(json.dumps(metadata))
    inference = Path(config["run_dir"]) / "inference/viewer-test"
    inference.mkdir(parents=True)
    (inference / "token_acts.npz").write_bytes(
        (Path(config.pop("inference_dir")) / "token_acts.npz").read_bytes()
    )
    config["shards_dir"] = str(shards)
    _, codes, _, images = load_source(config, "patch")
    assert codes.shape == (16, 3) and len(images) == 4


def test_cls_api_defaults_and_removed_mode(exported):
    from visualization.api import EditInput

    assert (
        EditInput(mode="cls", dimensions=[0], strength=0).method == "sae_gradient_delta"
    )
    root, _ = exported
    with TestClient(create_app(root)) as client:
        response = client.post(
            "/api/images/0/edit",
            json={
                "mode": "cls",
                "dimensions": [0],
                "strength": 0,
                "method": "sae",
            },
        )
        assert response.status_code == 422
