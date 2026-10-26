"""Render square images and patch maps in the encoder's spatial coordinates."""

import io
import math

import numpy as np
from PIL import Image


def png_bytes(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def display_image(path, size):
    with Image.open(path) as image:
        return image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)


def overlay(image, values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    side = math.isqrt(len(values))
    if side * side != len(values) or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected a finite square patch map")
    maximum = float(np.abs(values).max())
    heat = np.clip(values / maximum, 0, 1) if maximum else np.zeros_like(values)
    heat = Image.fromarray(np.uint8(heat.reshape(side, side) * 255)).resize(
        image.size, Image.Resampling.NEAREST
    )
    strength = np.asarray(heat, dtype=np.float32)[..., None] / 255 * 0.55
    pixels = np.asarray(image, dtype=np.float32)
    pixels = pixels * (1 - strength) + np.array([240, 76, 35]) * strength
    return Image.fromarray(np.uint8(np.clip(pixels, 0, 255)))


def tensor_image(tensor):
    pixels = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray(np.uint8(pixels * 255))
