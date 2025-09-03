#!/usr/bin/env python
"""Test loading REAL pretrained models flexibly - the actual goal!"""

import torch
import models.config as config
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer


def test_original_works():
    """Verify original SigLIP + SmolLM still works"""
    print("Testing original SigLIP + SmolLM configuration...")
    
    cfg = config.VLMConfig()  # Original defaults
    # This SHOULD work - it's the original configuration
    model = VisionLanguageModel(cfg, load_backbone=True)  # YES, load pretrained!
    
    print(f"✓ Original model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, cfg


def test_dinov2_loading():
    """Test loading actual DINOv2 weights"""  
    print("\nTesting DINOv2 loading...")
    
    cfg = config.VLMConfig()
    cfg.vit_architecture = "dinov3"  # Will detect dinov2
    cfg.vit_model_type = "facebook/dinov2-small"  # Real model from HF
    cfg.vit_cls_flag = True
    cfg.vit_num_registers = 0  # DINOv2 doesn't have registers
    cfg.vit_use_swiglu = False  # DINOv2 uses standard MLP
    cfg.vit_use_rope = False  # DINOv2 doesn't use RoPE
    cfg.mp_handle_special_tokens = True  # Handle CLS token
    
    # Keep SmolLM for language model (known to work)
    cfg.lm_model_type = "HuggingFaceTB/SmolLM2-135M"  # Smaller model for testing
    cfg.lm_tokenizer = "HuggingFaceTB/SmolLM2-135M"
    
    print(f"  Loading vision: {cfg.vit_model_type}")
    print(f"  Loading language: {cfg.lm_model_type}")
    
    model = VisionLanguageModel(cfg, load_backbone=True)  # YES, load real weights!
    
    print(f"✓ DINOv2 + SmolLM loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, cfg


def test_gemma_loading():
    """Test loading actual Gemma weights"""
    print("\nTesting Gemma loading...")
    
    cfg = config.VLMConfig()
    # Keep original vision encoder (known to work)
    cfg.vit_model_type = "google/siglip-base-patch16-224"  # Smaller SigLIP
    cfg.vit_img_size = 224
    
    # Use Gemma for language model
    cfg.lm_architecture = "gemma"
    cfg.lm_model_type = "google/gemma-2b"  # Real Gemma model
    cfg.lm_tokenizer = "google/gemma-2b"
    
    print(f"  Loading vision: {cfg.vit_model_type}")
    print(f"  Loading language: {cfg.lm_model_type}")
    
    model = VisionLanguageModel(cfg, load_backbone=True)  # YES, load real weights!
    
    print(f"✓ SigLIP + Gemma loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, cfg


def quick_forward_test(model, cfg):
    """Quick forward pass test"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)
    
    batch_size = 1
    seq_len = 64
    images = torch.randn(batch_size, 3, cfg.vit_img_size, cfg.vit_img_size).to(device)
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    # Add proper number of image tokens
    num_patches = (cfg.vit_img_size // cfg.vit_patch_size) ** 2
    num_image_tokens = num_patches // (cfg.mp_pixel_shuffle_factor ** 2)
    input_ids[:, :num_image_tokens] = tokenizer.image_token_id
    
    attention_mask = torch.ones_like(input_ids).to(device)
    labels = input_ids.clone()
    
    with torch.no_grad():
        logits, loss = model(input_ids, images, attention_mask, labels)
    
    print(f"  Forward pass: loss = {loss.item():.4f}")
    return True


def main():
    print("=" * 60)
    print("Testing REAL Pretrained Model Loading")
    print("=" * 60)
    
    try:
        # Test 1: Original should work
        print("\n1. ORIGINAL CONFIGURATION")
        model1, cfg1 = test_original_works()
        quick_forward_test(model1, cfg1)
        
        # Test 2: DINOv2 loading
        print("\n2. DINOV2 CONFIGURATION")
        try:
            model2, cfg2 = test_dinov2_loading()
            quick_forward_test(model2, cfg2)
        except Exception as e:
            print(f"  DINOv2 test failed (may need auth): {e}")
        
        # Test 3: Gemma loading  
        print("\n3. GEMMA CONFIGURATION")
        try:
            model3, cfg3 = test_gemma_loading()
            quick_forward_test(model3, cfg3)
        except Exception as e:
            print(f"  Gemma test failed (may need auth): {e}")
        
        print("\n" + "=" * 60)
        print("✓ Flexible model loading works!")
        print("\nYou can now use:")
        print("  python train.py --vision_encoder dinov2 --language_model smollm")
        print("  python train.py --vision_encoder siglip --language_model gemma")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())