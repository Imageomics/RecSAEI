"""Serve exported feature activations and optional RAEv2 interventions."""

import argparse
import base64
import json
from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock
from typing import Literal

import numpy as np
import yaml
from core.selection import manual_dimensions
from core.selection.activation import select_activation
from core.selection.class_specific import select_class
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from scipy import sparse

from .rendering import display_image, overlay, png_bytes, tensor_image


class EditInput(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    mode: Literal["patch", "cls"]
    dimensions: list[int] = Field(min_length=1)
    strength: float
    formula: Literal["steering_delta", "direct_decode"] = "steering_delta"
    mask: Literal["global", "high_activation", "manual"] = "global"
    patches: list[int] | None = None
    method: Literal["sae_gradient_delta", "dense"] = "sae_gradient_delta"
    source_dimensions: int = Field(default=8, ge=1)
    selection_mode: Literal["activation_weighted", "sensitivity"] = (
        "activation_weighted"
    )


def create_app(export_dir, *, runtime_config=None, runtime_factory=None):
    root = Path(export_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    images = json.loads((root / "images.json").read_text())["items"]
    image_map = {row["image_id"]: row for row in images}
    data = {}
    for mode in manifest["modes"]:
        folder = root / mode
        data[mode] = {
            name: sparse.load_npz(folder / f"{name}.npz").tocsr()
            for name in ("token_acts", "mean", "max")
        }
        data[mode]["classes"] = {}
        if (folder / "classes.npz").exists():
            with np.load(folder / "classes.npz") as classes:
                data[mode]["classes"] = {key: classes[key] for key in classes}
    image_rows = {row["image_id"]: i for i, row in enumerate(images)}
    runtime = None
    lock = RLock()

    def live():
        nonlocal runtime
        if runtime_config is None:
            raise HTTPException(
                400, "Live interventions require a runtime configuration"
            )
        if runtime is None:
            for mode, info in manifest["modes"].items():
                checkpoint = runtime_config.get(f"{mode}_checkpoint")
                if (
                    checkpoint
                    and Path(checkpoint).resolve()
                    != Path(info["sae_checkpoint"]).resolve()
                ):
                    raise ValueError(
                        f"The {mode} runtime checkpoint differs from the exported SAE"
                    )
            from .runtime import LiveRuntime

            runtime = (runtime_factory or LiveRuntime)(runtime_config)
        return runtime

    def information(mode):
        if mode not in data:
            raise HTTPException(404, "Unknown token mode")
        return manifest["modes"][mode]

    def image_record(image_id):
        if image_id not in image_map:
            raise HTTPException(404, "Unknown image ID")
        return image_map[image_id]

    def codes_for(image_id, mode):
        image_record(image_id)
        row = image_rows[image_id]
        n = information(mode)["tokens_per_image"]
        return data[mode]["token_acts"][row * n : (row + 1) * n]

    @asynccontextmanager
    async def lifespan(app):
        if runtime_config is not None:
            live()
        yield

    app = FastAPI(title=manifest["title"], lifespan=lifespan)
    frontend = Path(__file__).parent / "frontend"
    app.mount("/_viewer", StaticFiles(directory=frontend), name="viewer")

    @app.exception_handler(FileNotFoundError)
    @app.exception_handler(ValueError)
    async def invalid_request(request, exc):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/")
    def index():
        return FileResponse(frontend / "index.html")

    @app.get("/api/manifest")
    def get_manifest():
        return {**manifest, "live_enabled": runtime_config is not None}

    @app.get("/api/images")
    def get_images():
        return {
            "items": [
                {key: value for key, value in row.items() if key != "source_path"}
                for row in images
            ]
        }

    @app.get("/api/dimensions")
    def dimensions(
        mode: str,
        method: Literal["activation", "manual", "class_specific"] = "activation",
        top_k: int = Query(20, ge=1, le=200),
        pooling: Literal["mean", "max"] = "mean",
        image_id: str | None = None,
        class_name: str | None = None,
        dim: list[int] = Query(default=[]),
        patch: list[int] = Query(default=[]),
    ):
        info = information(mode)
        if method == "manual":
            items = manual_dimensions(dim, d_sae=info["d_sae"])
        elif method == "class_specific":
            items = select_class(data[mode]["classes"], class_name, top_k=top_k)
        elif image_id is not None:
            if patch and mode == "cls":
                raise ValueError("CLS activations have no spatial patch indices")
            items = select_activation(
                codes_for(image_id, mode),
                top_k=top_k,
                pooling=pooling,
                patches=patch or None,
            )
        else:
            if patch:
                raise ValueError("Select an image before selecting patches")
            # Across images use a mean of the chosen per-image pooling statistic.
            items = select_activation(data[mode][pooling], top_k=top_k, pooling="mean")
            for item in items:
                item["method"] = f"dataset_mean_of_{pooling}"
        return {"items": items}

    @app.get("/api/dimensions/{dimension}")
    def gallery(
        dimension: int,
        mode: str,
        top_k: int = Query(6, ge=1, le=100),
        pooling: Literal["mean", "max"] = "max",
    ):
        manual_dimensions([dimension], d_sae=information(mode)["d_sae"])
        scores = data[mode][pooling][:, [dimension]].toarray().reshape(-1)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return {
            "dim": dimension,
            "items": [
                dict(
                    image_id=images[i]["image_id"],
                    score=float(scores[i]),
                    class_name=images[i]["class_name"],
                )
                for i in order
            ],
        }

    @app.get("/api/images/{image_id}/image")
    def image(image_id: str, mode: str):
        info = information(mode)
        return Response(
            png_bytes(
                display_image(image_record(image_id)["source_path"], info["image_size"])
            ),
            media_type="image/png",
        )

    @app.get("/api/images/{image_id}/values")
    def values(image_id: str, mode: str, dim: int):
        manual_dimensions([dim], d_sae=information(mode)["d_sae"])
        return {
            "values": codes_for(image_id, mode)[:, [dim]]
            .toarray()
            .reshape(-1)
            .tolist(),
            "mode": mode,
        }

    @app.get("/api/images/{image_id}/overlay")
    def activation_overlay(image_id: str, mode: str, dim: int):
        info = information(mode)
        manual_dimensions([dim], d_sae=info["d_sae"])
        if mode == "cls":
            with lock:
                activations = live().patch_attribution(
                    image_record(image_id)["source_path"], target_dimension=dim
                )
        else:
            activations = codes_for(image_id, mode)[:, [dim]].toarray()
        image = display_image(image_record(image_id)["source_path"], info["image_size"])
        return Response(png_bytes(overlay(image, activations)), media_type="image/png")

    @app.get("/api/images/{image_id}/reconstruction")
    def reconstruction(image_id: str):
        with lock:
            result = live().reconstruct(image_record(image_id)["source_path"])
        return Response(png_bytes(tensor_image(result)), media_type="image/png")

    @app.post("/api/images/{image_id}/edit")
    def edit(image_id: str, request: EditInput):
        manual_dimensions(request.dimensions, d_sae=information(request.mode)["d_sae"])
        if request.mode == "patch" and request.mask == "manual" and not request.patches:
            raise ValueError("Manual editing requires patch indices")
        if request.patches is not None:
            if request.mode != "patch":
                raise ValueError("Manual patch editing is available in patch mode")
            count = information("patch")["tokens_per_image"]
            if len(set(request.patches)) != len(request.patches) or any(
                p < 0 or p >= count for p in request.patches
            ):
                raise ValueError("Invalid manual patch indices")
        with lock:
            result = live().edit(image_record(image_id)["source_path"], request)
        return {
            "image": "data:image/png;base64,"
            + base64.b64encode(png_bytes(tensor_image(result.image))).decode(),
            "details": result.details,
        }

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    import uvicorn

    uvicorn.run(
        create_app(args.export_dir, runtime_config=config.get("runtime")),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
