#!/usr/bin/env python
"""Debug DINOv3 token count."""

import torch

from models.config import get_dinov3_gemma_config
from models.modality_projector import ModalityProjector
from models.vision_transformer import ViT

config = get_dinov3_gemma_config()
print(f"Config: {config.vit_architecture}")

# Create vision encoder
vision = ViT.from_pretrained(config)

# Create modality projector
mp = ModalityProjector(config)

# Process single 224x224 image
image = torch.randn(1, 3, 224, 224)
print(f"Input image shape: {image.shape}")

with torch.no_grad():
    # Get vision features
    features = vision(image)
    print(f"Vision output shape: {features.shape}")
    print(f"  Tokens: {features.shape[1]} (CLS + registers + patches)")
    
    # Project through modality projector
    projected = mp(features)
    print(f"Projected shape: {projected.shape}")
    print(f"  Final tokens: {projected.shape[1]}")
    
    # Calculate expected
    num_patches = 14 * 14  # 224/16 = 14
    print("\nExpected:")
    print(f"  Patches: {num_patches}")
    print("  CLS: 1")
    print(f"  Registers: {config.vit_num_registers}")
    print(f"  Total before projection: {num_patches + 1 + config.vit_num_registers}")
    print(f"  After PixelShuffle (factor {config.mp_pixelshuffle_factor}): {(num_patches + 1 + config.vit_num_registers) // (config.mp_pixelshuffle_factor ** 2)}")
    print(f"  But special tokens removed: {(num_patches) // (config.mp_pixelshuffle_factor ** 2)}")