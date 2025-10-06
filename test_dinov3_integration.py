#!/usr/bin/env python
"""Quick test to verify DINOv3 integration works."""

import torch
from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel

# Test both DINOv3 variants
import sys
encoder_type = sys.argv[1] if len(sys.argv) > 1 else "dinov3-small"
print(f"Testing {encoder_type} integration...")

cfg = VLMConfig()
cfg.vision_encoder_type = encoder_type
cfg.lm_model_type = "HuggingFaceTB/SmolLM2-135M"

# Create model
print(f"Creating model with {cfg.vision_encoder_type}...")
model = VisionLanguageModel(cfg, load_backbone=True)
model.eval()

# Create dummy inputs with image tokens
batch_size = 1
# Create input with image tokens (token ID for <|image|> is the first image token)
image_token_id = model.tokenizer.convert_tokens_to_ids("<|image|>")
print(f"Image token ID: {image_token_id}")

# Create sequence with 64 image tokens (as expected by modality projector)
num_image_tokens = 64
dummy_input_ids = torch.cat([
    torch.full((batch_size, num_image_tokens), image_token_id),  # Image tokens
    torch.randint(0, 49000, (batch_size, 36))  # Text tokens
], dim=1)
seq_len = dummy_input_ids.shape[1]
dummy_images = torch.randn(1, 3, 512, 512)  # Single image

# Test forward pass
print("Testing forward pass...")
with torch.no_grad():
    logits, loss = model(
        input_ids=dummy_input_ids,
        images=dummy_images,
        attention_mask=torch.ones(batch_size, seq_len),
        targets=None
    )

print(f"✓ Forward pass successful!")
print(f"  Output shape: {logits.shape}")
print(f"  Vision encoder params: {model.vision_encoder.get_total_params():,}")
print(f"  Trainable params: {model.vision_encoder.get_num_trainable_params():,}")

# Test freezing
model.vision_encoder.freeze()
print(f"✓ After freezing: {model.vision_encoder.get_num_trainable_params():,} trainable params")

print("\nAll tests passed! DINOv3-small integration is working.")