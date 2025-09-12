#!/usr/bin/env python3
"""
Test script for validating the grid abstraction layer with both DINOv3 and SigLIP.
"""

import sys
from dataclasses import dataclass

# Test the grid abstraction module
from models.grid_abstraction import DINOv3Grid, GridFactory, SigLIPGrid, normalize_grids


@dataclass
class MockConfig:
    """Mock config for testing."""

    vit_architecture: str = "siglip"
    mp_pixel_shuffle_factor: int = 2
    mp_image_token_length: int = 64


def test_dinov3_grid():
    """Test DINOv3 grid functionality."""
    print("Testing DINOv3Grid...")

    # Test creation from dict
    raw_data = {"HpWp": (16, 24), "GhGw": (8, 12)}
    grid = DINOv3Grid.from_raw(raw_data)

    assert grid.Hp == 16 and grid.Wp == 24
    assert grid.Gh == 8 and grid.Gw == 12
    assert grid.get_modality_projector_dims(2) == (16, 24)
    assert grid.get_final_grid_dims() == (8, 12)
    assert grid.get_token_count() == 96

    # Test dict with only GhGw
    raw_data2 = {"GhGw": (4, 6)}
    grid2 = DINOv3Grid.from_raw(raw_data2, pixel_shuffle_factor=2)
    assert grid2.Hp == 8 and grid2.Wp == 12
    assert grid2.Gh == 4 and grid2.Gw == 6

    print("✓ DINOv3Grid tests passed")


def test_siglip_grid():
    """Test SigLIP grid functionality."""
    print("Testing SigLIPGrid...")

    # Test creation from tuple
    raw_data = (2, 3)
    grid = SigLIPGrid.from_raw(raw_data, tokens_per_tile=64)

    assert grid.rows == 2 and grid.cols == 3
    assert grid.get_modality_projector_dims(2) == (4, 6)
    assert grid.get_final_grid_dims() == (2, 3)
    assert grid.get_token_count() == 384  # 2*3*64

    # Test creation from dict
    raw_data2 = {"grid": (1, 2), "tokens_per_tile": 32}
    grid2 = SigLIPGrid.from_raw(raw_data2)
    assert grid2.rows == 1 and grid2.cols == 2
    assert grid2.tokens_per_tile == 32
    assert grid2.get_token_count() == 64

    print("✓ SigLIPGrid tests passed")


def test_grid_factory():
    """Test GridFactory functionality."""
    print("Testing GridFactory...")

    # Test with DINOv3 config
    config = MockConfig(vit_architecture="dinov3")

    # DINOv3 dict format
    grid1 = GridFactory.create_from_raw({"GhGw": (4, 4), "HpWp": (8, 8)}, config)
    assert isinstance(grid1, DINOv3Grid)
    assert grid1.Gh == 4 and grid1.Gw == 4

    # Tuple format with DINOv3 config (should convert)
    grid2 = GridFactory.create_from_raw((4, 4), config)
    assert isinstance(grid2, DINOv3Grid)
    assert grid2.Gh == 4 and grid2.Gw == 4
    assert grid2.Hp == 8 and grid2.Wp == 8  # Computed from shuffle factor

    # Test with SigLIP config
    config.vit_architecture = "siglip"

    # Tuple format
    grid3 = GridFactory.create_from_raw((2, 2), config)
    assert isinstance(grid3, SigLIPGrid)
    assert grid3.rows == 2 and grid3.cols == 2

    # None input
    grid4 = GridFactory.create_from_raw(None, config)
    assert isinstance(grid4, SigLIPGrid)
    assert grid4.rows == 1 and grid4.cols == 1

    print("✓ GridFactory tests passed")


def test_normalize_grids():
    """Test normalize_grids functionality."""
    print("Testing normalize_grids...")

    config = MockConfig(vit_architecture="siglip")

    # Mixed input formats
    raw_grids = [
        (2, 2),  # SigLIP tuple
        {"grid": (1, 3), "tokens_per_tile": 32},  # SigLIP dict
        None,  # Will create default
    ]

    normalized = normalize_grids(raw_grids, config)

    assert len(normalized) == 3
    assert all(isinstance(g, SigLIPGrid) for g in normalized)
    assert normalized[0].rows == 2 and normalized[0].cols == 2
    assert normalized[1].rows == 1 and normalized[1].cols == 3
    assert normalized[2].rows == 1 and normalized[2].cols == 1  # Default

    # Test with DINOv3
    config.vit_architecture = "dinov3"
    raw_grids2 = [
        {"GhGw": (4, 6)},
        (2, 3),  # Will be converted to DINOv3
    ]

    normalized2 = normalize_grids(raw_grids2, config)
    assert len(normalized2) == 2
    assert all(isinstance(g, DINOv3Grid) for g in normalized2)

    print("✓ normalize_grids tests passed")


def test_mixed_formats():
    """Test handling of mixed formats in a realistic scenario."""
    print("Testing mixed format handling...")

    # Simulate what happens when switching between models
    config = MockConfig()

    # Start with SigLIP format
    config.vit_architecture = "siglip"
    siglip_data = (2, 3)
    grid1 = GridFactory.create_from_raw(siglip_data, config)

    # Switch to DINOv3 but still have SigLIP-style data
    config.vit_architecture = "dinov3"
    grid2 = GridFactory.create_from_raw(siglip_data, config)

    # Both should work correctly
    assert isinstance(grid1, SigLIPGrid)
    assert isinstance(grid2, DINOv3Grid)

    # Test round-trip through dict
    dict1 = grid1.to_dict()
    dict2 = grid2.to_dict()

    assert dict1["type"] == "siglip"
    assert dict2["type"] == "dinov3"

    print("✓ Mixed format tests passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Grid Abstraction Layer")
    print("=" * 60)

    try:
        test_dinov3_grid()
        test_siglip_grid()
        test_grid_factory()
        test_normalize_grids()
        test_mixed_formats()

        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
