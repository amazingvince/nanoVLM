#!/usr/bin/env python
"""Test that DINOv3 config overrides are properly applied."""

from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel


def test_config_overrides():
    """Test that config overrides are applied to DINOv3 model."""
    print("Testing DINOv3 config override functionality...\n")

    # Create config with custom values
    cfg = VLMConfig()
    cfg.vision_encoder_type = "dinov3-small"
    cfg.vit_layerscale_value = 0.5  # Override from default 1.0
    cfg.vit_drop_path_rate = 0.1  # Override from default 0.0
    cfg.vit_pos_embed_shift = 0.1  # Add position augmentation
    cfg.vit_pos_embed_jitter = 1.5
    cfg.vit_pos_embed_rescale = 3.0

    print("Creating model with config overrides:")
    print(f"  LayerScale: {cfg.vit_layerscale_value}")
    print(f"  DropPath: {cfg.vit_drop_path_rate}")
    print(
        f"  Position augmentation: shift={cfg.vit_pos_embed_shift}, jitter={cfg.vit_pos_embed_jitter}, rescale={cfg.vit_pos_embed_rescale}"
    )
    print()

    # Create model
    model = VisionLanguageModel(cfg, load_backbone=True)

    # Check that values were applied
    encoder = model.vision_encoder
    model_config = encoder.model.config

    print("Verifying config values in loaded model:")

    # Check LayerScale
    actual_layerscale = getattr(model_config, "layerscale_value", None)
    expected_layerscale = cfg.vit_layerscale_value
    status = "✓" if actual_layerscale == expected_layerscale else "✗"
    print(
        f"  LayerScale: {actual_layerscale} (expected {expected_layerscale}) {status}"
    )

    # Check DropPath
    actual_drop_path = getattr(model_config, "drop_path_rate", None)
    expected_drop_path = cfg.vit_drop_path_rate
    status = "✓" if actual_drop_path == expected_drop_path else "✗"
    print(f"  DropPath: {actual_drop_path} (expected {expected_drop_path}) {status}")

    # Check position augmentation
    actual_shift = getattr(model_config, "pos_embed_shift", None)
    expected_shift = cfg.vit_pos_embed_shift
    status = "✓" if actual_shift == expected_shift else "✗"
    print(f"  Position shift: {actual_shift} (expected {expected_shift}) {status}")

    actual_jitter = getattr(model_config, "pos_embed_jitter", None)
    expected_jitter = cfg.vit_pos_embed_jitter
    status = "✓" if actual_jitter == expected_jitter else "✗"
    print(f"  Position jitter: {actual_jitter} (expected {expected_jitter}) {status}")

    actual_rescale = getattr(model_config, "pos_embed_rescale", None)
    expected_rescale = cfg.vit_pos_embed_rescale
    status = "✓" if actual_rescale == expected_rescale else "✗"
    print(
        f"  Position rescale: {actual_rescale} (expected {expected_rescale}) {status}"
    )

    # Test training mode
    print("\nTesting training mode activation:")
    model.vision_encoder.train()  # Should log that augmentations are active

    print("\nTesting eval mode:")
    model.vision_encoder.eval()  # Should disable augmentations

    print("\nTesting freeze mode:")
    model.vision_encoder.freeze()  # Should disable augmentations and dropout

    print("\n✅ Config override test complete!")


if __name__ == "__main__":
    test_config_overrides()
