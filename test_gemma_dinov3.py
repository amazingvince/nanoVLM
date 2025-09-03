#!/usr/bin/env python
"""Test script to demonstrate Gemma + DINOv3 architecture support"""

import torch
import models.config as config
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer


def test_gemma_dinov3_architecture():
    """Test that we can create and run a model with the new architectures"""
    print("=" * 60)
    print("Testing DINOv3-style Vision Encoder + Gemma-style LM")
    print("=" * 60)
    
    # Create a config with new architecture features
    # Note: Using small models and not loading pretrained weights to avoid download issues
    cfg = config.VLMConfig()
    
    # Enable new architecture features
    cfg.vit_architecture = "dinov3"
    cfg.vit_use_swiglu = True  # Use SwiGLU FFN
    cfg.vit_use_rope = True  # Use RoPE
    cfg.vit_cls_flag = True  # Use CLS token
    cfg.vit_num_registers = 4  # Use register tokens
    cfg.mp_handle_special_tokens = True  # Handle special tokens in modality projector
    
    # Configure smaller dimensions for testing
    cfg.vit_hidden_dim = 384
    cfg.vit_inter_dim = 1536
    cfg.vit_n_heads = 6
    cfg.vit_n_blocks = 4  # Fewer blocks for testing
    cfg.vit_img_size = 224
    cfg.vit_patch_size = 14
    
    # Update modality projector for new dimensions
    cfg.mp_pixel_shuffle_factor = 2
    # Calculate correct mp_image_token_length
    # patches = (224 / 14)^2 = 16^2 = 256
    # after pixel shuffle with factor 2: 256 / 4 = 64
    cfg.mp_image_token_length = 64
    
    # Configure language model
    cfg.lm_architecture = "gemma"
    cfg.lm_hidden_dim = 768
    cfg.lm_inter_dim = 2048
    cfg.lm_n_heads = 12
    cfg.lm_n_kv_heads = 3
    cfg.lm_n_blocks = 8  # Fewer blocks for testing
    cfg.lm_vocab_size = 32000  # Smaller vocab for testing
    cfg.lm_base_vocab_size = 32000
    
    # Don't load pretrained weights
    cfg.vlm_load_backbone_weights = False
    
    print("\nConfiguration:")
    print(f"  Vision: {cfg.vit_architecture} with SwiGLU={cfg.vit_use_swiglu}, RoPE={cfg.vit_use_rope}")
    print(f"  Vision: CLS={cfg.vit_cls_flag}, Registers={cfg.vit_num_registers}")
    print(f"  Language: {cfg.lm_architecture} architecture")
    print(f"  Modality Projector: handle_special_tokens={cfg.mp_handle_special_tokens}")
    
    # Create model
    print("\nInitializing model...")
    model = VisionLanguageModel(cfg, load_backbone=False)  # Don't load pretrained weights
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters")
    
    # Check submodules
    print("\nModel components:")
    print(f"  Vision encoder: {model.vision_encoder.__class__.__name__}")
    print(f"  Language model: {model.decoder.__class__.__name__}")
    print(f"  Modality projector: {model.MP.__class__.__name__}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    # Create dummy inputs
    tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)
    
    batch_size = 2
    seq_len = 128
    images = torch.randn(batch_size, 3, cfg.vit_img_size, cfg.vit_img_size).to(device)
    
    # Create input with image tokens
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    input_ids[:, :cfg.mp_image_token_length] = tokenizer.image_token_id
    
    attention_mask = torch.ones_like(input_ids).to(device)
    labels = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    
    with torch.no_grad():
        # Test vision encoder
        vision_out = model.vision_encoder(images)
        print(f"  Vision encoder output shape: {vision_out.shape}")
        
        # Expected: [batch_size, 1 + 4 + 196, 384] = [2, 201, 384]
        # (1 CLS + 4 registers + 196 patches for 224x224 with patch_size=14)
        expected_seq = 1 + cfg.vit_num_registers + (cfg.vit_img_size // cfg.vit_patch_size) ** 2
        assert vision_out.shape == (batch_size, expected_seq, cfg.vit_hidden_dim)
        
        # Test modality projector
        mp_out = model.MP(vision_out)
        print(f"  Modality projector output shape: {mp_out.shape}")
        
        # After removing special tokens and pixel shuffle
        assert mp_out.shape == (batch_size, cfg.mp_image_token_length, cfg.lm_hidden_dim)
        
        # Test full forward pass
        logits, loss = model(input_ids, images, attention_mask, labels)
        print(f"  Full model output shape: {logits.shape}")
        print(f"  Loss value: {loss.item():.4f}")
    
    print("\n✓ All architecture features working correctly!")
    return True


def main():
    try:
        success = test_gemma_dinov3_architecture()
        print("\n" + "=" * 60)
        if success:
            print("SUCCESS: DINOv3 + Gemma architecture support validated!")
            print("\nYou can now train models with:")
            print("  python train.py --vision_encoder dinov3 --language_model gemma")
            print("  python train.py --use_preset dinov3_gemma")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())