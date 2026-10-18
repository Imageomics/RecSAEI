from collections.abc import Callable

import beartype
import torch
from jaxtyping import Float, jaxtyped
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import v2
from transformers import SiglipVisionModel

from . import models

HF_ENCODER_IMPLEMENTATION = "huggingface-transformers-siglip2"


@jaxtyped(typechecker=beartype.beartype)
class Vit(torch.nn.Module, models.Transformer):
    """SigLIP 2 adapter matching the RAEv2 encoder implementation."""

    family: str = "siglip2"
    patch_size: int = 16
    has_cls_token: bool = False
    encoder_implementation: str = HF_ENCODER_IMPLEMENTATION

    def __init__(self, ckpt: str):
        super().__init__()
        self._ckpt = ckpt
        self.model = SiglipVisionModel.from_pretrained(ckpt)

    @property
    def ckpt(self) -> str:
        return self._ckpt

    @staticmethod
    def _input_size_from_patch_count(content_tokens_per_example: int) -> int:
        if content_tokens_per_example <= 0:
            raise ValueError("content_tokens_per_example must be positive")
        patches_per_side = int(content_tokens_per_example**0.5)
        if patches_per_side * patches_per_side != content_tokens_per_example:
            raise ValueError(
                "Expected square SigLIP 2 patch grid, got "
                f"{content_tokens_per_example} patches"
            )
        return patches_per_side * Vit.patch_size

    @staticmethod
    def make_transforms(
        ckpt: str,
        content_tokens_per_example: int,
        *,
        image_resize_mode: str = "square",
    ) -> tuple[Callable, Callable | None]:
        del ckpt
        if image_resize_mode != "square":
            raise ValueError(
                "RAEv2 SigLIP 2 shards require image_resize_mode='square'"
            )
        input_size = Vit._input_size_from_patch_count(content_tokens_per_example)
        normalize = v2.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        def transform(image: Image.Image) -> Tensor:
            tensor = v2.functional.to_image(image)
            tensor = v2.functional.to_dtype(tensor, torch.float32, scale=True)
            tensor = normalize(tensor)
            return F.interpolate(
                tensor.unsqueeze(0),
                size=(input_size, input_size),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)

        return transform, None

    @staticmethod
    def make_resize(
        ckpt: str,
        content_tokens_per_example: int = 256,
        *,
        scale: float = 1.0,
        resample: Image.Resampling = Image.Resampling.LANCZOS,
    ) -> Callable[[Image.Image], Image.Image]:
        del ckpt
        input_size = Vit._input_size_from_patch_count(content_tokens_per_example)

        def resize(image: Image.Image) -> Image.Image:
            size = int(input_size * scale)
            return image.resize((size, size), resample=resample)

        return resize

    def get_residuals(self) -> list[torch.nn.Module]:
        return list(self.model.encoder.layers)

    @staticmethod
    def get_token_i(content_tokens_per_example: int) -> slice:
        Vit._input_size_from_patch_count(content_tokens_per_example)
        return slice(None, None, None)

    def forward(
        self, batch: Float[Tensor, "batch 3 width height"], **kwargs
    ) -> Float[Tensor, "batch patches dim"]:
        if kwargs:
            raise ValueError(f"Unexpected SigLIP 2 forward arguments: {sorted(kwargs)}")
        return self.model(batch).last_hidden_state
