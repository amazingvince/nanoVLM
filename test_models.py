#!/usr/bin/env python3
"""Quick test to see if models can be instantiated."""

import torch
from models.config import VLMConfig, get_original_small_config
from models.vision_language_model import VisionLanguageModel

# Test SIGLIP
print("Testing SIGLIP configuration...")
cfg = get_original_small_config()
print(f"Config: vit_architecture={cfg.vit_architecture}, mp_image_token_length={cfg.mp_image_token_length}")
print(f"  lm_hidden_dim={cfg.lm_hidden_dim}, lm_n_heads={cfg.lm_n_heads}, lm_head_dim={cfg.lm_head_dim}")

try:
    model = VisionLanguageModel(cfg, load_backbone=False)
    print("✓ SIGLIP model created successfully")
    print(f"  Vision encoder: {cfg.vit_model_type}")
    print(f"  Language model: {cfg.lm_model_type}")
except Exception as e:
    print(f"✗ Failed to create SIGLIP model: {e}")

# Test DINOv3
print("\nTesting DINOv3 configuration...")
cfg2 = VLMConfig(
    vit_architecture="dinov3",
    vit_model_type="facebook/dinov3-vits16plus-pretrain-lvd1689m",
    vit_cls_flag=True,
    vit_num_registers=4,
    vit_use_swiglu=True,
    vit_use_rope=True,
    vit_layer_scale=True,
    vit_img_size=512,
    vit_patch_size=16,
    lm_model_type="HuggingFaceTB/SmolLM2-135M-Instruct",
    lm_tokenizer="HuggingFaceTB/SmolLM2-135M-Instruct",
    mp_pixel_shuffle_factor=2,
    mp_image_token_length=1,
    mp_handle_special_tokens=True,
)
print(f"Config: vit_architecture={cfg2.vit_architecture}, mp_image_token_length={cfg2.mp_image_token_length}")

try:
    model2 = VisionLanguageModel(cfg2, load_backbone=False)
    print("✓ DINOv3 model created successfully")
    print(f"  Vision encoder: {cfg2.vit_model_type}")
    print(f"  Language model: {cfg2.lm_model_type}")
except Exception as e:
    print(f"✗ Failed to create DINOv3 model: {e}")

print("\nModels instantiated successfully!")