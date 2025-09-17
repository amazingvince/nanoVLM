import math
from typing import Tuple, Union

import torch
from einops import rearrange
from PIL import Image
from torchvision.transforms.functional import InterpolationMode, resize


class DynamicResize(torch.nn.Module):
    """
    Resize so that:
      * the longer side ≤ `max_side_len` **and** is divisible by `patch_size`
      * the shorter side keeps aspect ratio and is also divisible by `patch_size`
    Optionally forbids up-scaling.

    Works on PIL Images, (C, H, W) tensors, or (B, C, H, W) tensors.
    Returns the same type it receives.
    """

    def __init__(
        self,
        patch_size: int,
        max_side_len: int,
        resize_to_max_side_len: bool = False,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    ) -> None:
        super().__init__()
        self.p = int(patch_size)
        self.m = int(max_side_len)
        self.interpolation = interpolation
        print(f"Resize to max side len: {resize_to_max_side_len}")
        self.resize_to_max_side_len = resize_to_max_side_len

    # ------------------------------------------------------------
    def _get_new_hw(self, h: int, w: int) -> Tuple[int, int]:
        """Compute target (h, w) divisible by patch_size."""
        long, short = (w, h) if w >= h else (h, w)

        # 1) upscale long side
        target_long = (
            self.m
            if self.resize_to_max_side_len
            else min(self.m, math.ceil(long / self.p) * self.p)
        )

        # 2) scale factor
        scale = target_long / long

        # 3) compute short side with ceil → never undershoot
        target_short = math.ceil(short * scale / self.p) * self.p
        target_short = max(target_short, self.p)  # just in case

        return (target_short, target_long) if w >= h else (target_long, target_short)

    # ------------------------------------------------------------
    def forward(self, img: Union[Image.Image, torch.Tensor]):
        if isinstance(img, Image.Image):
            w, h = img.size
            new_h, new_w = self._get_new_hw(h, w)
            return resize(img, [new_h, new_w], interpolation=self.interpolation)

        if not torch.is_tensor(img):
            raise TypeError(
                f"DynamicResize expects a PIL Image or a torch.Tensor; got {type(img)}"
            )

        # tensor path ---------------------------------------------------------
        batched = img.ndim == 4
        if img.ndim not in (3, 4):
            raise ValueError(
                f"Tensor input must have shape (C,H,W) or (B,C,H,W); got {img.shape}"
            )

        # operate batch-wise
        imgs = img if batched else img.unsqueeze(0)
        _, _, h, w = imgs.shape
        new_h, new_w = self._get_new_hw(h, w)
        out = resize(imgs, [new_h, new_w], interpolation=self.interpolation)

        return out if batched else out.squeeze(0)


class SplitImage(torch.nn.Module):
    """Split (B, C, H, W) image tensor into square patches.

    Returns:
        patches: (B·n_h·n_w, C, patch_size, patch_size)
        grid:    (n_h, n_w)  - number of patches along H and W
    """

    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.p = patch_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if x.ndim == 3:  # add batch dim if missing
            x = x.unsqueeze(0)

        b, c, h, w = x.shape
        if h % self.p or w % self.p:
            raise ValueError(
                f"Image size {(h, w)} not divisible by patch_size {self.p}"
            )

        n_h, n_w = h // self.p, w // self.p
        patches = rearrange(
            x, "b c (nh ph) (nw pw) -> (b nh nw) c ph pw", ph=self.p, pw=self.p
        )
        return patches, (n_h, n_w)


class GlobalAndSplitImages(torch.nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.p = patch_size
        self.splitter = SplitImage(patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if x.ndim == 3:
            x = x.unsqueeze(0)

        patches, grid = self.splitter(x)

        if grid == (1, 1):
            return patches, grid  # Dont add global patch if there is only one patch

        global_patch = resize(x, [self.p, self.p])
        return torch.cat([global_patch, patches], dim=0), grid
