import math
from typing import Tuple, Union

import torch
from einops import rearrange
from PIL import Image
from torchvision.transforms.functional import InterpolationMode, resize


class DynamicResize(torch.nn.Module):
    """Dynamic image resizing to ensure dimensions are divisible by patch size.

    :param patch_size: Size of patches for vision transformer
    :param max_side_len: Maximum allowed side length
    :param resize_to_max_side_len: Whether to always resize to max_side_len
    :param interpolation: Interpolation mode for resizing
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
        """Compute target dimensions divisible by patch size.

        :param h: Original height
        :param w: Original width
        :return: Tuple of (new_height, new_width)
        """
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
    def forward(
        self, img: Union[Image.Image, torch.Tensor]
    ) -> Union[Image.Image, torch.Tensor]:
        """Resize input image maintaining aspect ratio.

        :param img: Input PIL Image or tensor
        :return: Resized image in same format as input
        """
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
    """Split image tensor into square patches for vision transformer.

    :param patch_size: Size of each square patch
    """

    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.p = patch_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Split input tensor into patches.

        :param x: Input tensor [batch, channels, height, width]
        :return: Tuple of (patches [B*n_h*n_w, C, p, p], grid (n_h, n_w))
        """
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
    """Split images into patches with optional fixed-size windowing.

    Supports two modes:
    1. Standard mode (use_windowing=False): Original behavior for SigLIP
       - Images split into variable number of patches based on size
       - Global context added for multi-patch images

    2. Windowed mode (use_windowing=True): For encoders needing fixed token counts
       - Images split into fixed-size windows (e.g., 256×256)
       - Each window produces exactly mp_image_token_length tokens
       - Enables high-resolution support for DINOv3

    :param patch_size: Size of each square patch (typically 16)
    :param use_windowing: Whether to use fixed-size window splitting
    :param window_size: Size of each window in pixels (only for windowed mode)
    """

    def __init__(
        self, patch_size: int, use_windowing: bool = False, window_size: int = 256
    ):
        super().__init__()
        self.p = patch_size
        self.use_windowing = use_windowing
        self.window_size = window_size

        # For standard mode, reuse the original SplitImage
        if not use_windowing:
            self.splitter = SplitImage(patch_size)
        else:
            # Validate window size is divisible by patch size
            if window_size % patch_size != 0:
                raise ValueError(
                    f"Window size {window_size} must be divisible by patch size {patch_size}"
                )
            print(f"Window mode enabled: {window_size}×{window_size}px windows")

    def _standard_forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Original SigLIP behavior: variable patches with optional global context."""
        patches, grid = self.splitter(x)

        if grid == (1, 1):
            return patches, grid  # Don't add global patch if there is only one patch

        # Add global context for multi-patch images
        global_patch = resize(x, [self.p, self.p])
        return torch.cat([global_patch, patches], dim=0), grid

    def _windowed_forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Window-based splitting for fixed token counts."""
        b, c, h, w = x.shape

        # If image fits in one window, just validate and return
        if h <= self.window_size and w <= self.window_size:
            if h % self.p or w % self.p:
                raise ValueError(
                    f"Image size {(h, w)} not divisible by patch_size {self.p}"
                )
            return x, (1, 1)

        # Calculate number of windows needed
        n_windows_h = math.ceil(h / self.window_size)
        n_windows_w = math.ceil(w / self.window_size)

        # Pad image to be divisible by window size
        pad_h = n_windows_h * self.window_size - h
        pad_w = n_windows_w * self.window_size - w

        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        # Extract windows
        windows = []

        # First add global context (downsampled full image)
        global_window = resize(x, [self.window_size, self.window_size])
        windows.append(global_window)

        # Extract each window
        for i in range(n_windows_h):
            for j in range(n_windows_w):
                h_start = i * self.window_size
                w_start = j * self.window_size
                window = x[
                    :,
                    :,
                    h_start : h_start + self.window_size,
                    w_start : w_start + self.window_size,
                ]
                windows.append(window)

        # Stack all windows
        all_windows = torch.cat(windows, dim=0)
        return all_windows, (n_windows_h, n_windows_w)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Split image based on configured mode.

        :param x: Input tensor [batch, channels, height, width]
        :return: Tuple of (patches/windows, grid dimensions)
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)

        if self.use_windowing:
            return self._windowed_forward(x)
        else:
            return self._standard_forward(x)
