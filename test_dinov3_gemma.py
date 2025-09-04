#!/usr/bin/env python
"""Quick test to verify DINOv3+Gemma3 loads correctly."""

import torch

from models.config import get_dinov3_gemma_config
from models.vision_language_model import VisionLanguageModel

# Create config
config = get_dinov3_gemma_config()
print("Config created: DINOv3 + Gemma-3-270M")
print(f"  Vision: {config.vit_architecture} - {config.vit_model_type}")
print(f"  Language: {config.lm_architecture} - {config.lm_model_type}")
print(f"  Vision features: RoPE={config.vit_use_rope}, Registers={config.vit_num_registers}, LayerScale={config.vit_layer_scale}")

# Create model
print("\nCreating model...")
model = VisionLanguageModel(config)
print("Model created successfully!")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass with dummy data
print("\nTesting forward pass...")
from data.processors import get_tokenizer

tokenizer = get_tokenizer(config.lm_tokenizer, config.vlm_extra_tokens, config.lm_chat_template)
batch_size = 2
seq_len = 128
vocab_size = config.lm_vocab_size

# Create inputs with image tokens
# For DINOv3: the modality projector outputs 49 tokens for a 224x224 image
num_image_tokens = 49
text = "This is a test. " + tokenizer.image_token * num_image_tokens + " What do you see?"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
input_ids = input_ids.repeat(batch_size, 1)

# Single 224x224 image for DINOv3
images = [torch.randn(3, 224, 224) for _ in range(batch_size)]

# Forward pass
with torch.no_grad():
    try:
        output = model(input_ids, images)
        if isinstance(output, tuple):
            logits = output[0]  # Extract logits from tuple
        else:
            logits = output
        print("Forward pass successful!")
        print(f"  Output shape: {logits.shape}")
        print(f"  Expected: ({batch_size}, {input_ids.shape[1]}, {vocab_size})")
        
        # Verify output shape
        assert logits.shape[0] == batch_size, "Batch size mismatch!"
        assert logits.shape[2] == vocab_size, "Vocab size mismatch!"
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()