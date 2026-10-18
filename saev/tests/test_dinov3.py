import importlib.util
import pathlib
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from saev.data import dinov3
from saev.data.shards import RecordedTransformer


@pytest.mark.parametrize("n_patches", [196, 256])
def test_preprocessing_matches_raev2(n_patches):
    loader_path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "RAEv2/src/encoders/models/dinov3_loader.py"
    )
    spec = importlib.util.spec_from_file_location("raev2_dinov3_loader", loader_path)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    pixels = np.random.default_rng(0).integers(0, 256, (93, 157, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    transform, sample_transform = dinov3.Vit.make_transforms(
        "unused", n_patches, image_resize_mode="square"
    )
    side = int(n_patches**0.5) * 16
    expected = loader.make_dinov3_transform(side)(
        torch.from_numpy(pixels).permute(2, 0, 1).float()
    )
    actual = transform(image)
    assert sample_transform is None
    assert actual.shape == (3, side, side)
    torch.testing.assert_close(actual, expected)


def test_preprocessing_requires_square_patch_grid():
    with pytest.raises(ValueError, match="perfect-square"):
        dinov3.Vit.make_transforms("unused", 255, image_resize_mode="square")
    with pytest.raises(ValueError, match="square"):
        dinov3.Vit.make_transforms("unused", 256)


class FakeOfficialEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_storage_tokens = 4
        self.storage_tokens = torch.nn.Parameter(torch.zeros(1, 4, 3))
        self.blocks = torch.nn.ModuleList([torch.nn.Identity()])

    def forward_features(self, images):
        # CLS, four internal registers, and four patch tokens.
        tokens = torch.arange(27, dtype=images.dtype).reshape(1, 9, 3)
        tokens = self.blocks[0](tokens.expand(images.shape[0], -1, -1))
        return {
            "x_norm_clstoken": tokens[:, 0],
            "x_norm_patchtokens": tokens[:, 5:],
        }


@pytest.mark.parametrize("cls_token", [False, True])
def test_official_loader_and_recording_skip_internal_registers(monkeypatch, cls_token):
    official = FakeOfficialEncoder()
    hub_load = Mock(return_value=official)
    monkeypatch.setattr(torch.hub, "load", hub_load)
    ckpt = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    model = dinov3.Vit(ckpt)
    hub_load.assert_called_once_with(
        dinov3.DINOV3_HUB_REF,
        "dinov3_vits16",
        source="github",
        trust_repo=True,
        skip_validation=True,
        weights=ckpt,
    )
    assert not model.training
    assert not official.training
    recorder = RecordedTransformer(model, 4, cls_token, [0])
    output, activations = recorder(torch.zeros(2, 3, 32, 32))
    assert output.shape == (2, 5, 3)
    expected = output if cls_token else output[:, 1:]
    torch.testing.assert_close(activations[:, 0], expected)
