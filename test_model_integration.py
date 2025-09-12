#!/usr/bin/env python3
"""
Integration test for the grid abstraction with actual model components.
Tests both SigLIP and DINOv3 configurations.
"""

import sys

import numpy as np
from PIL import Image

from data.processors import get_image_processor

# Import model components
from models.config import VLMConfig
from models.grid_abstraction import GridFactory, normalize_grids


def create_test_image(size=(512, 384)):
    """Create a test image."""
    # Create a simple gradient image
    arr = np.zeros((*size, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, size[0])[:, None]  # Red gradient
    arr[:, :, 1] = np.linspace(0, 255, size[1])[None, :]  # Green gradient
    arr[:, :, 2] = 128  # Constant blue
    return Image.fromarray(arr)


def test_siglip_processing():
    """Test SigLIP image processing with grid abstraction."""
    print("\n" + "=" * 60)
    print("Testing SigLIP Configuration")
    print("=" * 60)

    # Create SigLIP config
    config = VLMConfig()
    config.vit_architecture = "siglip"
    config.vit_image_size = 224
    config.mp_pixel_shuffle_factor = 2
    config.mp_image_token_length = 64

    # Create processor
    processor = get_image_processor(
        max_img_size=512,
        splitted_image_size=224,
        single_image_mode=False,
        vit_patch_size=16,
        pixel_shuffle_factor=config.mp_pixel_shuffle_factor,
        allow_upscale=True,
    )

    # Process test image
    test_img = create_test_image((512, 384))
    processed_img, grid_data = processor(test_img)

    print(f"Processed image shape: {processed_img.shape}")
    print(f"Grid data (raw): {grid_data}")

    # Test grid abstraction
    grid = GridFactory.create_from_raw(grid_data, config)
    print(f"Grid type: {type(grid).__name__}")
    print(f"Grid dimensions: {grid.get_final_grid_dims()}")
    print(
        f"Modality projector dims: {grid.get_modality_projector_dims(config.mp_pixel_shuffle_factor)}"
    )
    print(f"Token count: {grid.get_token_count()}")

    # Test normalization
    normalized = normalize_grids([grid_data], config)
    assert len(normalized) == 1
    assert normalized[0].get_final_grid_dims() == grid.get_final_grid_dims()

    print("✅ SigLIP processing test passed")
    return True


def test_dinov3_processing():
    """Test DINOv3 image processing with grid abstraction."""
    print("\n" + "=" * 60)
    print("Testing DINOv3 Configuration")
    print("=" * 60)

    # Create DINOv3 config
    config = VLMConfig()
    config.vit_architecture = "dinov3"
    config.vit_image_size = 512
    config.vit_patch_size = 16
    config.mp_pixel_shuffle_factor = 2
    config.mp_image_token_length = 1  # DINOv3 uses 1 token per grid cell

    # Create processor
    processor = get_image_processor(
        max_img_size=1024,
        splitted_image_size=512,
        single_image_mode=True,  # DINOv3 uses single image mode
        vit_patch_size=config.vit_patch_size,
        pixel_shuffle_factor=config.mp_pixel_shuffle_factor,
        allow_upscale=True,
    )

    # Process test images with different aspect ratios
    test_images = [
        create_test_image((512, 512)),  # Square
        create_test_image((768, 512)),  # Wide
        create_test_image((512, 768)),  # Tall
    ]

    for i, test_img in enumerate(test_images):
        print(f"\nTest image {i + 1}: {test_img.size}")
        processed_img, grid_data = processor(test_img)

        print(f"  Processed shape: {processed_img.shape}")
        print(f"  Grid data: {grid_data}")

        # Test grid abstraction
        grid = GridFactory.create_from_raw(grid_data, config)
        print(f"  Grid type: {type(grid).__name__}")
        print(f"  Pre-shuffle dims (Hp, Wp): {grid.Hp, grid.Wp}")
        print(f"  Post-shuffle dims (Gh, Gw): {grid.Gh, grid.Gw}")
        print(f"  Token count: {grid.get_token_count()}")

        # Verify divisibility
        assert grid.Hp % config.mp_pixel_shuffle_factor == 0
        assert grid.Wp % config.mp_pixel_shuffle_factor == 0
        assert grid.Hp // config.mp_pixel_shuffle_factor == grid.Gh
        assert grid.Wp // config.mp_pixel_shuffle_factor == grid.Gw

    print("\n✅ DINOv3 processing test passed")
    return True


def test_mixed_batch():
    """Test handling mixed grid formats in a batch."""
    print("\n" + "=" * 60)
    print("Testing Mixed Batch Processing")
    print("=" * 60)

    config = VLMConfig()
    config.mp_pixel_shuffle_factor = 2

    # Simulate mixed batch with different grid formats
    mixed_grids = [
        (2, 2),  # SigLIP tuple
        {"GhGw": (4, 6), "HpWp": (8, 12)},  # DINOv3 dict
        (3, 3),  # Another SigLIP
        {"grid": (1, 4), "tokens_per_tile": 32},  # SigLIP dict format
    ]

    # Test with SigLIP config
    config.vit_architecture = "siglip"
    config.mp_image_token_length = 64
    normalized_siglip = normalize_grids(mixed_grids, config)

    print("SigLIP mode normalization:")
    for i, grid in enumerate(normalized_siglip):
        print(f"  Grid {i}: {type(grid).__name__} - {grid.get_final_grid_dims()}")

    # Test with DINOv3 config
    config.vit_architecture = "dinov3"
    config.mp_image_token_length = 1
    normalized_dinov3 = normalize_grids(mixed_grids, config)

    print("\nDINOv3 mode normalization:")
    for i, grid in enumerate(normalized_dinov3):
        print(f"  Grid {i}: {type(grid).__name__} - {grid.get_final_grid_dims()}")

    print("\n✅ Mixed batch test passed")
    return True


def test_backward_compatibility():
    """Test that the new abstraction maintains backward compatibility."""
    print("\n" + "=" * 60)
    print("Testing Backward Compatibility")
    print("=" * 60)

    config = VLMConfig()
    config.mp_pixel_shuffle_factor = 2

    # Test old format handling
    old_formats = [
        (2, 3),  # Old SigLIP tuple
        [(1, 1), (2, 2)],  # List of tuples
    ]

    for fmt in old_formats:
        try:
            if isinstance(fmt, list):
                normalized = normalize_grids(fmt, config)
                print(f"✓ Successfully handled list format: {len(normalized)} grids")
            else:
                grid = GridFactory.create_from_raw(fmt, config)
                print(f"✓ Successfully handled format: {fmt} -> {type(grid).__name__}")
        except Exception as e:
            print(f"✗ Failed to handle format {fmt}: {e}")
            return False

    print("\n✅ Backward compatibility test passed")
    return True


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("Grid Abstraction Integration Tests")
    print("=" * 60)

    tests = [
        ("SigLIP Processing", test_siglip_processing),
        ("DINOv3 Processing", test_dinov3_processing),
        ("Mixed Batch", test_mixed_batch),
        ("Backward Compatibility", test_backward_compatibility),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} failed with error: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All integration tests passed!")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
