"""Small cached bridge from browser requests to RAEv2 interventions."""

from collections import OrderedDict

import torch
from core.intervention import (
    CLSRequest,
    GlobalMask,
    HighActivationMask,
    ManualPatchMask,
    PatchEditRequest,
    RAEv2Backend,
    ScaleTarget,
    intervene_cls,
    intervene_patch,
)


class LiveRuntime:
    def __init__(self, config):
        self.backend = RAEv2Backend.from_config(
            config["stage1_config"],
            device=config.get("device", "cuda"),
            raev2_root=config.get("raev2_root"),
            dinov3_checkpoint_dir=config.get("dinov3_checkpoint_dir"),
        )
        self.patch_sae = (
            self.backend.load_sae(config["patch_checkpoint"])
            if config.get("patch_checkpoint")
            else None
        )
        self.cls_sae = (
            self.backend.load_sae(config["cls_checkpoint"], token_mode="cls")
            if config.get("cls_checkpoint")
            else None
        )
        self.cache = OrderedDict()
        self.cache_size = int(config.get("cache_size", 8))
        if self.cache_size < 1:
            raise ValueError("cache_size must be positive")

    def image(self, path):
        if path not in self.cache:
            self.cache[path] = self.backend.load_image_tensor(path)
        self.cache.move_to_end(path)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return self.cache[path]

    def reconstruct(self, path):
        return self.backend.reconstruct_tensor(self.image(path))[0]

    def edit(self, path, request):
        image = self.image(path)
        if request.mode == "patch":
            if self.patch_sae is None:
                raise ValueError("Patch editing requires a patch SAE checkpoint")
            dims = tuple(request.dimensions)
            if request.patches:
                mask = ManualPatchMask(tuple(request.patches))
            elif request.mask == "high_activation":
                mask = HighActivationMask(dims)
            else:
                mask = GlobalMask()
            return intervene_patch(
                backend=self.backend,
                sae=self.patch_sae,
                image_tensor=image,
                request=PatchEditRequest(
                    ScaleTarget(dims, request.strength), request.formula, mask
                ),
            )
        if self.cls_sae is None or len(request.dimensions) != 1:
            raise ValueError(
                "CLS steering requires a CLS SAE and exactly one target dimension"
            )
        return intervene_cls(
            backend=self.backend,
            cls_sae=self.cls_sae,
            patch_sae=self.patch_sae,
            image_tensor=image,
            request=CLSRequest(
                request.dimensions[0],
                request.strength,
                method=request.method,
                source_dimensions=request.source_dimensions,
                selection_mode=request.selection_mode,
            ),
        )

    def patch_attribution(self, path, *, target_dimension):
        """Spatial gradient magnitude of a CLS feature with respect to patch tokens."""
        if self.cls_sae is None:
            raise ValueError("CLS attribution requires a CLS SAE checkpoint")
        with torch.enable_grad():
            full = self.backend.capture_final_block_input(self.image(path))
            codes = self.cls_sae.encode(
                self.backend.run_final_block(full, normalize=False)[:, 0]
            ).f_x
            if not 0 <= target_dimension < codes.shape[1]:
                raise ValueError("Invalid CLS target dimension")
            gradient = torch.autograd.grad(codes[0, target_dimension], full)[0]
        return gradient[0, self.backend.patch_slice].norm(dim=-1).detach().cpu().tolist()
