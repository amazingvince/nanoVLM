#!/usr/bin/env python
"""Test script for DINOv3 + Gemma integration"""

import torch

import models.config as config
from models.vision_language_model import VisionLanguageModel


def test_config_creation():
    """Test that DINOv3 + Gemma configuration can be created"""
    print("Testing configuration creation...")
    
    # Test default config
    default_cfg = config.VLMConfig()
    assert default_cfg.vit_architecture == "siglip"
    assert default_cfg.lm_architecture == "llama"
    print("✓ Default configuration created successfully")
    
    # Test DINOv3 + Gemma preset
    dinov3_gemma_cfg = config.get_dinov3_gemma_config()
    assert dinov3_gemma_cfg.vit_architecture == "dinov3"
    assert dinov3_gemma_cfg.lm_architecture == "gemma"
    assert dinov3_gemma_cfg.vit_cls_flag == True
    assert dinov3_gemma_cfg.vit_num_registers == 4
    assert dinov3_gemma_cfg.mp_handle_special_tokens == True
    print("✓ DINOv3 + Gemma configuration created successfully")
    
    return dinov3_gemma_cfg


def test_model_initialization(cfg):
    """Test that the model can be initialized without loading weights"""
    print("\nTesting model initialization...")
    
    # Set to not load pretrained weights for testing
    cfg.vlm_load_backbone_weights = False
    
    try:
        model = VisionLanguageModel(cfg)
        print("✓ Model initialized successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
        
        # Check submodules
        assert hasattr(model, 'vision_encoder')
        assert hasattr(model, 'language_model')
        assert hasattr(model, 'modality_projector')
        print("✓ All required submodules present")
        
        return model
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        raise


def test_forward_pass(model, cfg):
    """Test a forward pass with dummy data"""
    print("\nTesting forward pass...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 128
    img_size = cfg.vit_img_size
    
    # Dummy image
    images = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    # Dummy input tokens
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    
    # Dummy attention mask
    attention_mask = torch.ones_like(input_ids).to(device)
    
    # Dummy labels
    labels = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    
    try:
        with torch.no_grad():
            logits, loss = model(input_ids, [images, images], attention_mask, labels)
        
        assert logits is not None
        assert loss is not None
        assert logits.shape[0] == batch_size
        print(f"✓ Forward pass successful")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Loss value: {loss.item():.4f}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise


def test_vision_encoder_components(cfg):
    """Test individual vision encoder components"""
    print("\nTesting vision encoder components...")
    
    from models.vision_transformer import VisionRotaryEmbedding, ViTSwiGLUFFN
    
    # Test RoPE
    rope = VisionRotaryEmbedding(dim=64, base=10000.0)
    cos, sin = rope(100)
    assert cos.shape == (100, 64)
    assert sin.shape == (100, 64)
    print("✓ Vision RoPE works correctly")
    
    # Test SwiGLU FFN
    swiglu_cfg = type('cfg', (), {
        'vit_hidden_dim': 384,
        'vit_inter_dim': 1536,
        'vit_dropout': 0.0
    })()
    swiglu = ViTSwiGLUFFN(swiglu_cfg)
    x = torch.randn(2, 10, 384)
    out = swiglu(x)
    assert out.shape == (2, 10, 384)
    print("✓ SwiGLU FFN works correctly")


def main():
    print("=" * 50)
    print("Testing DINOv3 + Gemma Integration")
    print("=" * 50)
    
    # Test configuration
    cfg = test_config_creation()
    
    # Test vision encoder components
    test_vision_encoder_components(cfg)
    
    # Test model initialization
    model = test_model_initialization(cfg)
    
    # Test forward pass
    test_forward_pass(model, cfg)
    
    print("\n" + "=" * 50)
    print("✓ All tests passed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()