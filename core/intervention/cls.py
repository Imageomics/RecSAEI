"""Steer raw final-block CLS SAE features through penultimate patch tokens.

Both methods add a direction normalized jointly over the masked patches.
Strength multiplies the Frobenius norm of ALL original patch tokens, excluding
CLS/register tokens. There is no per-patch averaging or masked-patch rescaling.
The decoder always receives the resulting final patch tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch

from .backend import RAEv2Backend
from .common import InterventionResult


@dataclass(frozen=True)
class CLSRequest:
    target_dimension: int
    strength: float
    method: Literal["sae_gradient_delta", "dense"] = "sae_gradient_delta"
    source_dimensions: int = 8
    mass_fraction: float = 0.8
    min_patches: int = 1
    selection_mode: Literal["activation_weighted", "sensitivity"] = (
        "activation_weighted"
    )


def replace_patches(
    full: torch.Tensor, patches: torch.Tensor, patch_slice: slice
) -> torch.Tensor:
    # Concatenation retains the attribution graph and excludes every special token.
    return torch.cat(
        (full[:, : patch_slice.start], patches[None], full[:, patch_slice.stop :]), 1
    )


def dense_direction(gradient, mask):
    direction = gradient.detach() * mask[:, None]
    norm = direction.norm()
    if not torch.isfinite(direction).all() or norm <= 1e-12:
        raise ValueError("Dense patch gradient is zero or non-finite")
    return direction / norm


def sae_gradient_direction(codes, gradient, decoder_weights, mask, edit):
    source_dimensions = int(edit.get("source_dimensions", 8))
    if source_dimensions < 1:
        raise ValueError("SAE gradient delta requires at least one dimension")
    # Attribution selects a global SAE subspace. Once selected, every dimension
    # in that subspace is weighted at every patch in the *same* raw-gradient
    # spatial mask used by dense steering. This deliberately avoids a second,
    # SAE-specific top-patches restriction.
    relevance = (codes * gradient).masked_fill(~mask[:, None], 0)
    relevance = relevance.masked_fill(codes <= 0, 0)
    scores = relevance.abs().sum(0)
    selection_mode = edit.get("selection_mode", "activation_weighted")
    if selection_mode not in ("activation_weighted", "sensitivity"):
        raise ValueError("selection_mode must be activation_weighted or sensitivity")
    if selection_mode == "sensitivity":
        # Match the effective count of activation-weighted selection, but rank
        # the entire dictionary, including features inactive on the mask.
        source_dimensions = min(source_dimensions, int((scores > 0).sum()))
        scores = gradient.masked_fill(~mask[:, None], 0).abs().sum(0)
    order = torch.argsort(scores, descending=True, stable=True)
    # A requested K can exceed the activation x gradient support on a particular
    # image. Cap at the nonzero support rather than filling the direction with
    # zero-attribution features. The caller records both requested and effective
    # K so saturated sweep points remain explicit.
    contributing_dimensions = [dim for dim in order.tolist() if scores[dim] > 0]
    dimensions = contributing_dimensions[:source_dimensions]
    if not dimensions:
        raise ValueError("SAE gradient delta has no contributing dimensions")
    weights = torch.zeros_like(gradient)
    weights[:, dimensions] = gradient.detach()[:, dimensions]
    weights.masked_fill_(~mask[:, None], 0)
    vectors = weights @ decoder_weights.detach()
    norm = vectors.norm()
    if not torch.isfinite(vectors).all() or norm <= 1e-12:
        raise ValueError("Combined SAE delta is zero or non-finite")
    contributions = []
    for dim in dimensions:
        contributions.append(
            {
                "source_dimension": dim,
                "score": float(scores[dim]),
                "masked_patch_count": int(mask.sum()),
                "nonzero_weight_patch_count": int((weights[:, dim] != 0).sum()),
                "gradient_l2": float(weights[:, dim].norm()),
                "patches": [
                    {
                        "patch": patch,
                        "activation": float(codes[patch, dim]),
                        "gradient": float(gradient[patch, dim]),
                        "attribution": float(relevance[patch, dim]),
                        "weight": float(weights[patch, dim]),
                        "normalized_weight": float(weights[patch, dim] / norm),
                    }
                    for patch in torch.where(mask)[0].tolist()
                ],
            }
        )
    return vectors / norm, weights, contributions


def _target_patch_mask(
    scores: torch.Tensor,
    cfg: dict[str, Any] | None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    mask_cfg = dict(cfg or {})
    mode = str(mask_cfg.get("mode", "mass")).strip().lower()
    if mode != "mass":
        raise ValueError(f"Unknown target_patch_mask mode: {mode}")
    mass_fraction = float(mask_cfg.get("mass_fraction", 0.8))
    if not 0.0 < mass_fraction <= 1.0:
        raise ValueError("target_patch_mask.mass_fraction must be in (0, 1]")
    min_patches = int(mask_cfg.get("min_patches", 1))
    if min_patches < 0:
        raise ValueError("target_patch_mask.min_patches must be non-negative")

    flat = scores.reshape(-1).float()
    mask = torch.zeros_like(flat, dtype=torch.bool)
    total = flat.clamp_min(0).sum()
    if float(total.item()) <= 0.0 or int(flat.numel()) == 0:
        return mask, {
            "target_patch_mask_count": 0,
            "target_patch_mask_mass": 0.0,
            "target_patch_mask_total_mass": float(total.item()),
        }

    values, indices = torch.sort(flat.clamp_min(0), descending=True)
    cumulative = torch.cumsum(values, dim=0)
    target_mass = total * mass_fraction
    count = int(torch.searchsorted(cumulative, target_mass, right=False).item()) + 1
    count = max(count, min_patches)
    count = min(count, int(flat.numel()))
    selected = indices[:count]
    mask[selected] = True
    selected_mass = flat[selected].clamp_min(0).sum()
    return mask, {
        "target_patch_mask_count": int(count),
        "target_patch_mask_mass": float(selected_mass.item()),
        "target_patch_mask_total_mass": float(total.item()),
    }


def intervene_cls(
    *,
    backend: RAEv2Backend,
    cls_sae: torch.nn.Module,
    image_tensor: torch.Tensor,
    request: CLSRequest,
    patch_sae: torch.nn.Module | None = None,
) -> InterventionResult:
    """Build one frozen local steering direction and render the requested strength.

    Target SAE inputs are raw final-block CLS states, before final encoder norm.
    Source SAE inputs are raw penultimate patches. CLS/register inputs stay fixed.
    An active target is required to define the local attribution direction.
    """
    if not backend.model_has_cls_token or backend.block_index != backend.num_blocks - 2:
        raise ValueError(
            "CLS steering requires CLS tokens and the penultimate editable block"
        )
    if request.method not in ("sae_gradient_delta", "dense"):
        raise ValueError(f"Unknown CLS steering method: {request.method}")
    if not math.isfinite(request.strength):
        raise ValueError("CLS strength must be finite")
    if request.method != "dense" and patch_sae is None:
        raise ValueError("This CLS method requires a patch SAE")
    full = backend.capture_final_block_input(image_tensor)
    patches = full[0, backend.patch_slice]
    baseline = cls_sae.encode(backend.run_final_block(full, normalize=False)[:, 0])
    dim = request.target_dimension
    if not 0 <= dim < baseline.f_x.shape[-1]:
        raise ValueError("CLS target dimension is out of range")
    if baseline.f_x[0, dim] <= 0:
        raise ValueError("CLS target must be active on the original image")
    gradient = torch.autograd.grad(baseline.f_x[0, dim], full)[0][
        0, backend.patch_slice
    ]
    mask, details = _target_patch_mask(
        gradient.detach().norm(dim=-1),
        {
            "mass_fraction": request.mass_fraction,
            "min_patches": request.min_patches,
        },
    )
    contributions = []
    if request.method == "dense":
        direction = dense_direction(gradient, mask)
    else:
        with torch.no_grad():
            codes = patch_sae.encode(patches.detach()).f_x.detach()
        code_gradient = gradient @ patch_sae.W_dec.detach().T
        direction, weights, contributions = sae_gradient_direction(
            codes,
            code_gradient,
            patch_sae.W_dec,
            mask,
            {
                "source_dimensions": request.source_dimensions,
                "selection_mode": request.selection_mode,
            },
        )
        with torch.no_grad():
            affine_delta = (
                patch_sae.decode(weights)[:, -1, :]
                - patch_sae.decode(torch.zeros_like(weights))[:, -1, :]
            )
            expected = weights @ patch_sae.W_dec
            tolerance = (
                8
                * torch.finfo(expected.dtype).eps
                * max(1.0, float(expected.abs().max()))
            )
            if not torch.allclose(affine_delta, expected, atol=tolerance, rtol=0):
                raise ValueError("SAE gradient delta requires an affine SAE decoder")
    # One image-wide budget, distributed only over the selected patch mask.
    scale = patches.detach().norm()
    edited = patches.detach() + request.strength * scale * direction
    if request.strength == 0:
        edited = patches.detach().clone()
    if not torch.isfinite(edited).all():
        raise ValueError("CLS intervention produced non-finite patch tokens")
    with torch.no_grad():
        target = cls_sae.encode(
            backend.run_final_block(
                replace_patches(full.detach(), edited, backend.patch_slice),
                normalize=False,
            )[:, 0]
        )
        image = backend.decode_target_tokens(
            image_tensor=image_tensor,
            editable_tokens=full.detach(),
            target_tokens=edited,
        )
    details.update(
        {
            "method": request.method,
            "target_dimension": dim,
            "strength": request.strength,
            "strength_units": "relative_patch_l2",
            "strength_reference": "all_original_patch_tokens",
            "original_patch_norm": float(scale),
            "baseline_activation": float(baseline.f_x[0, dim].detach()),
            "target_activation": float(target.f_x[0, dim]),
            "contributions": contributions,
        }
    )
    return InterventionResult(image, patches.detach(), edited.detach(), mask, details)
