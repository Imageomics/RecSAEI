"""RAEv2 intervention entry points for patch and CLS SAE features."""

from .backend import RAEv2Backend
from .cls import CLSRequest, intervene_cls
from .common import InterventionResult
from .patch import (
    ExplicitTarget,
    GlobalMask,
    HighActivationMask,
    HighActivationMaskConfig,
    ManualPatchMask,
    PatchEditRequest,
    ScaleTarget,
    intervene_patch,
)

__all__ = [
    "RAEv2Backend",
    "CLSRequest",
    "intervene_cls",
    "intervene_patch",
    "PatchEditRequest",
    "ExplicitTarget",
    "GlobalMask",
    "HighActivationMask",
    "HighActivationMaskConfig",
    "InterventionResult",
    "ManualPatchMask",
    "ScaleTarget",
]
