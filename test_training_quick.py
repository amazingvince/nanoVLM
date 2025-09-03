#!/usr/bin/env python
"""Quick test to ensure training can start with both configurations"""

import torch
import models.config as config
from models.vision_language_model import VisionLanguageModel


def test_original_config():
    """Test that the original configuration still works"""
    print("Testing original configuration...")
    
    vlm_cfg = config.VLMConfig()
    train_cfg = config.TrainConfig()
    
    # Don't load pretrained weights for testing
    vlm_cfg.vlm_load_backbone_weights = False
    
    model = VisionLanguageModel(vlm_cfg)
    print(f"✓ Original model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass with dummy data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Get the tokenizer to find image_token_id
    from data.processors import get_tokenizer
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)
    
    batch_size = 2
    seq_len = 128
    images = torch.randn(batch_size, 3, vlm_cfg.vit_img_size, vlm_cfg.vit_img_size).to(device)
    
    # Create input_ids with image tokens
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    # Insert image tokens at the beginning of each sequence
    num_image_tokens = vlm_cfg.mp_image_token_length
    input_ids[:, :num_image_tokens] = tokenizer.image_token_id
    
    attention_mask = torch.ones_like(input_ids).to(device)
    labels = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    
    with torch.no_grad():
        # Pass images as a tensor, not a list (one image per batch item)
        logits, loss = model(input_ids, images, attention_mask, labels)
    
    assert logits is not None
    assert loss is not None
    print(f"✓ Forward pass successful: loss = {loss.item():.4f}")
    return True


def test_modified_config():
    """Test the modified configuration with new architectures"""
    print("\nTesting modified configuration (with new features)...")
    
    vlm_cfg = config.VLMConfig()
    
    # When using SigLIP weights, we CANNOT use SwiGLU or other DINOv3 features
    # These features are architecture-specific and incompatible
    # Set to NOT load weights so we can test the new architecture features
    vlm_cfg.vlm_load_backbone_weights = False
    vlm_cfg.vit_use_swiglu = True  # Test SwiGLU (only works without pretrained weights)
    vlm_cfg.vit_use_rope = False  # Keep RoPE disabled for now
    vlm_cfg.mp_handle_special_tokens = False
    
    model = VisionLanguageModel(vlm_cfg)
    print(f"✓ Modified model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Get the tokenizer to find image_token_id
    from data.processors import get_tokenizer
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)
    
    batch_size = 2
    seq_len = 128
    images = torch.randn(batch_size, 3, vlm_cfg.vit_img_size, vlm_cfg.vit_img_size).to(device)
    
    # Create input_ids with image tokens
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    # Insert image tokens at the beginning of each sequence
    num_image_tokens = vlm_cfg.mp_image_token_length
    input_ids[:, :num_image_tokens] = tokenizer.image_token_id
    
    attention_mask = torch.ones_like(input_ids).to(device)
    labels = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    
    with torch.no_grad():
        # Pass images as a tensor, not a list (one image per batch item)
        logits, loss = model(input_ids, images, attention_mask, labels)
    
    assert logits is not None
    assert loss is not None
    print(f"✓ Forward pass successful: loss = {loss.item():.4f}")
    return True


def main():
    print("=" * 50)
    print("Testing nanoVLM with new architecture support")
    print("=" * 50)
    
    success = True
    
    try:
        test_original_config()
    except Exception as e:
        print(f"✗ Original config failed: {e}")
        success = False
    
    try:
        test_modified_config()
    except Exception as e:
        print(f"✗ Modified config failed: {e}")
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 50)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())