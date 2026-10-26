"""Patch SAE targets, spatial masks, edit formulas, and reconstruction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Mapping, Protocol, Sequence, TypeAlias

import torch

from .backend import RAEv2Backend
from .common import InterventionResult

TensorValues: TypeAlias = Sequence[float] | torch.Tensor
PatchEditFormula: TypeAlias = Literal["direct_decode", "steering_delta"]


@dataclass(frozen=True)
class HighActivationMaskConfig:
    mode: Literal["mass", "top_k"] = "mass"
    top_k: int = 16
    mass_fraction: float = 0.8
    min_patches: int = 4
    max_patches: int | None = None
    eps: float = 1e-8


@dataclass(frozen=True)
class ScaleTarget:
    dims: tuple[int, ...]
    scale: float = 1.0


@dataclass(frozen=True)
class ExplicitTarget:
    dims: tuple[int, ...]
    values_by_dim: Mapping[int, TensorValues]


@dataclass(frozen=True)
class GlobalMask:
    pass


@dataclass(frozen=True)
class HighActivationMask:
    dims: tuple[int, ...]
    config: HighActivationMaskConfig = field(default_factory=HighActivationMaskConfig)


@dataclass(frozen=True)
class ManualPatchMask:
    indices: tuple[int, ...]


PatchEditTarget: TypeAlias = ScaleTarget | ExplicitTarget
PatchEditMask: TypeAlias = GlobalMask | HighActivationMask | ManualPatchMask


@dataclass(frozen=True)
class PatchEditRequest:
    target: PatchEditTarget
    formula: PatchEditFormula
    mask: PatchEditMask


@dataclass
class PatchEditResult:
    selected_tokens: torch.Tensor
    target_tokens: torch.Tensor
    mask: torch.Tensor
    is_identity: bool = False


class SAEDecoder(Protocol):
    def decode(self, codes: torch.Tensor) -> torch.Tensor: ...


def manual_patch_mask(
    sae_codes: torch.Tensor, indices: tuple[int, ...] | list[int], *, cls_token: bool
) -> torch.Tensor:
    if not indices:
        raise ValueError("mask_indices must include at least one spatial patch index")
    spatial_start = int(cls_token)
    n_spatial = int(sae_codes.shape[0]) - spatial_start
    normalized: list[int] = []
    seen: set[int] = set()
    for raw_index in indices:
        index = int(raw_index)
        if index in seen:
            raise ValueError(f"Duplicate mask index: {index}")
        if index < 0 or index >= n_spatial:
            raise ValueError(
                f"mask index {index} is out of range for {n_spatial} spatial patches"
            )
        seen.add(index)
        normalized.append(index)
    mask = torch.zeros(sae_codes.shape[0], dtype=torch.bool, device=sae_codes.device)
    mask[
        torch.tensor(
            [spatial_start + index for index in normalized], device=mask.device
        )
    ] = True
    return mask


def _top_k_mask(
    spatial_codes: torch.Tensor, dims: tuple[int, ...], cfg: HighActivationMaskConfig
) -> torch.Tensor:
    if cfg.top_k <= 0:
        raise ValueError("top_k must be positive for high-activation edits")
    mask = torch.zeros(
        spatial_codes.shape[0], dtype=torch.bool, device=spatial_codes.device
    )
    k = min(int(cfg.top_k), int(spatial_codes.shape[0]))
    for dim in dims:
        mask[torch.topk(spatial_codes[:, dim], k=k).indices] = True
    return mask


def _mass_mask(
    spatial_codes: torch.Tensor, dims: tuple[int, ...], cfg: HighActivationMaskConfig
) -> torch.Tensor:
    if not 0.0 < cfg.mass_fraction <= 1.0:
        raise ValueError("mass_fraction must be in (0, 1]")
    if cfg.min_patches <= 0:
        raise ValueError("min_patches must be positive")
    if cfg.max_patches is not None and cfg.max_patches <= 0:
        raise ValueError("max_patches must be positive when set")
    mask = torch.zeros(
        spatial_codes.shape[0], dtype=torch.bool, device=spatial_codes.device
    )
    n_spatial = int(spatial_codes.shape[0])
    max_patches = (
        n_spatial if cfg.max_patches is None else min(int(cfg.max_patches), n_spatial)
    )
    min_patches = min(int(cfg.min_patches), max_patches)
    for dim in dims:
        activations = spatial_codes[:, dim].clamp_min(0)
        total = activations.sum()
        if float(total.item()) <= cfg.eps:
            continue
        values, indices = torch.sort(activations, descending=True)
        cutoff = cfg.mass_fraction * total
        n_selected = (
            int(
                torch.searchsorted(
                    torch.cumsum(values, dim=0), cutoff, right=False
                ).item()
            )
            + 1
        )
        n_selected = max(min_patches, min(n_selected, max_patches))
        mask[indices[:n_selected]] = True
    return mask


def high_activation_mask(
    sae_codes: torch.Tensor,
    dims: tuple[int, ...],
    cfg: HighActivationMaskConfig,
    *,
    cls_token: bool,
) -> torch.Tensor:
    mask = torch.zeros(sae_codes.shape[0], dtype=torch.bool, device=sae_codes.device)
    spatial_start = int(cls_token)
    spatial_codes = sae_codes[spatial_start:]
    if spatial_codes.numel() == 0:
        return mask
    if cfg.mode == "top_k":
        spatial_mask = _top_k_mask(spatial_codes, dims, cfg)
    elif cfg.mode == "mass":
        spatial_mask = _mass_mask(spatial_codes, dims, cfg)
    else:
        raise ValueError(f"Unknown high-activation mask mode: {cfg.mode}")
    mask[spatial_start:] = spatial_mask
    return mask


def _validate_dims(dims: tuple[int, ...], d_sae: int) -> None:
    if not dims:
        raise ValueError("At least one SAE dimension must be provided")
    if len(set(dims)) != len(dims):
        raise ValueError("SAE dimensions must be unique")
    invalid = [dim for dim in dims if dim < 0 or dim >= d_sae]
    if invalid:
        raise ValueError(f"Invalid SAE dimensions: {invalid}")


def _target_codes(
    source_codes: torch.Tensor, request: PatchEditRequest
) -> torch.Tensor:
    edited = source_codes.clone()
    target = request.target
    _validate_dims(target.dims, int(source_codes.shape[1]))
    if isinstance(target, ScaleTarget):
        if not math.isfinite(float(target.scale)):
            raise ValueError("scale must be finite")
        edited[:, list(target.dims)] *= float(target.scale)
        return edited
    if not isinstance(target, ExplicitTarget):
        raise TypeError(f"Unsupported edit target: {type(target)}")
    for dim, values in target.values_by_dim.items():
        if int(dim) not in target.dims:
            raise ValueError(
                f"Explicit SAE values provided for dim {dim}, but dims={list(target.dims)}"
            )
        tensor = torch.as_tensor(values, dtype=edited.dtype, device=edited.device)
        if tensor.ndim != 1 or int(tensor.shape[0]) != int(edited.shape[0]):
            raise ValueError(
                f"Expected {edited.shape[0]} SAE values for dim {dim}, got {tensor.numel()}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError("SAE values must be finite")
        edited[:, int(dim)] = tensor
    return edited


def apply_patch_edit(
    *,
    sae: SAEDecoder,
    selected_tokens: torch.Tensor,
    sae_codes: torch.Tensor,
    request: PatchEditRequest,
    cls_token: bool,
) -> PatchEditResult:
    """Apply a patch SAE-code target, token formula, and spatial mask."""
    if selected_tokens.ndim != 2 or sae_codes.ndim != 2:
        raise ValueError("selected_tokens and sae_codes must both be two-dimensional")
    if selected_tokens.shape[0] != sae_codes.shape[0]:
        raise ValueError("selected_tokens and sae_codes must have the same token count")
    edited_codes = _target_codes(sae_codes, request)
    is_identity = bool(torch.equal(edited_codes, sae_codes))
    if request.formula == "direct_decode":
        candidate_tokens = sae.decode(edited_codes)[:, -1, :]
    elif request.formula == "steering_delta":
        if is_identity:
            candidate_tokens = selected_tokens
        else:
            candidate_tokens = selected_tokens + (
                sae.decode(edited_codes)[:, -1, :] - sae.decode(sae_codes)[:, -1, :]
            )
    else:
        raise ValueError(f"Unknown edit formula: {request.formula}")

    if isinstance(request.mask, GlobalMask):
        mask = torch.ones(
            selected_tokens.shape[0], dtype=torch.bool, device=selected_tokens.device
        )
    elif isinstance(request.mask, HighActivationMask):
        _validate_dims(request.mask.dims, int(sae_codes.shape[1]))
        mask = high_activation_mask(
            sae_codes, request.mask.dims, request.mask.config, cls_token=cls_token
        )
    elif isinstance(request.mask, ManualPatchMask):
        mask = manual_patch_mask(sae_codes, request.mask.indices, cls_token=cls_token)
    else:
        raise TypeError(f"Unsupported edit mask: {type(request.mask)}")

    target_tokens = selected_tokens.clone()
    target_tokens[mask] = candidate_tokens[mask]
    return PatchEditResult(
        selected_tokens=selected_tokens,
        target_tokens=target_tokens,
        mask=mask,
        is_identity=is_identity and request.formula == "steering_delta",
    )


def intervene_patch(
    *,
    backend: RAEv2Backend,
    sae: torch.nn.Module,
    image_tensor: torch.Tensor,
    request: PatchEditRequest,
) -> InterventionResult:
    """Edit raw block-output patches, then run remaining encoder blocks and decode."""
    with torch.no_grad():
        full = backend.encode_editable_tokens(image_tensor)
        patches = backend.select_editable_tokens(full)[0]
        codes = sae.encode(patches).f_x
        edit = apply_patch_edit(
            selected_tokens=patches,
            sae_codes=codes,
            sae=sae,
            request=request,
            cls_token=False,
        )
        if not torch.isfinite(edit.target_tokens).all():
            raise ValueError("Patch intervention produced non-finite tokens")
        image = backend.decode_target_tokens(
            image_tensor=image_tensor,
            editable_tokens=full,
            target_tokens=edit.target_tokens,
        )
    return InterventionResult(
        image,
        patches.detach(),
        edit.target_tokens.detach(),
        edit.mask,
        {"formula": request.formula, "is_identity": edit.is_identity},
    )
