#!/usr/bin/env python3
"""Debug script to understand DINOv3 dimension issues."""

import torch
from PIL import Image
import numpy as np

# Create a test image
test_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

# Test DINOv3 processing
from data.processors import get_image_processor
from models.config import VLMConfig

# Create config for DINOv3
cfg = VLMConfig()
cfg.vit_architecture = "dinov3"
cfg.vit_img_size = 512
cfg.vit_patch_size = 16
cfg.mp_pixel_shuffle_factor = 4
cfg.max_img_size = 512

# Get transform
transform = get_image_processor(
    max_img_size=cfg.max_img_size,
    splitted_image_size=224,  # not used for dinov3
    single_image_mode=True,  # DINOv3 mode
    vit_patch_size=cfg.vit_patch_size,
    pixel_shuffle_factor=cfg.mp_pixel_shuffle_factor
)

# Process image
img_tensor, grid_info = transform(test_img)

print(f"Image tensor shape: {img_tensor.shape}")
print(f"Grid info: {grid_info}")

# Calculate what the vision encoder will produce
H, W = img_tensor.shape[1], img_tensor.shape[2]
Hp = H // cfg.vit_patch_size
Wp = W // cfg.vit_patch_size
print(f"Image size: {H}x{W}")
print(f"Patch grid: {Hp}x{Wp} = {Hp*Wp} patches")

# What the modality projector expects
if isinstance(grid_info, dict):
    if "Hp" in grid_info and "Wp" in grid_info:
        mp_Hp = grid_info["Hp"]
        mp_Wp = grid_info["Wp"]
        print(f"Modality projector expects: {mp_Hp}x{mp_Wp} = {mp_Hp*mp_Wp} patches")
        
        # Check if they match
        if Hp * Wp != mp_Hp * mp_Wp:
            print(f"ERROR: Mismatch! Vision encoder produces {Hp*Wp} but MP expects {mp_Hp*mp_Wp}")
        else:
            print("OK: Dimensions match")

# Test with smaller image
small_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
small_tensor, small_grid = transform(small_img)
print(f"\nSmall image tensor shape: {small_tensor.shape}")
print(f"Small grid info: {small_grid}")