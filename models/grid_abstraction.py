"""
Grid abstraction layer for unified handling of different vision encoder formats.

This module provides a clean abstraction for handling grid dimensions from different
vision encoders (DINOv3, SigLIP, etc.) without scattering if/else checks throughout
the codebase.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class ImageGrid(ABC):
    """Abstract base class for image grid representations."""

    @abstractmethod
    def get_modality_projector_dims(self, pixel_shuffle_factor: int) -> Tuple[int, int]:
        """Get dimensions for modality projector (pre-shuffle dimensions)."""
        pass

    @abstractmethod
    def get_final_grid_dims(self) -> Tuple[int, int]:
        """Get final grid dimensions after pixel shuffle."""
        pass

    @abstractmethod
    def get_token_count(self) -> int:
        """Get total number of tokens this grid will produce."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage/transmission."""
        pass

    @classmethod
    @abstractmethod
    def from_raw(cls, raw_data: Any, pixel_shuffle_factor: int = 2) -> "ImageGrid":
        """Create from raw data (dict, tuple, etc.)."""
        pass


@dataclass
class DINOv3Grid(ImageGrid):
    """Grid representation for DINOv3 with aspect-preserving resize."""

    Hp: int  # Pre-shuffle height in patches
    Wp: int  # Pre-shuffle width in patches
    Gh: int  # Post-shuffle height (grid height)
    Gw: int  # Post-shuffle width (grid width)

    def get_modality_projector_dims(self, pixel_shuffle_factor: int) -> Tuple[int, int]:
        """DINOv3 stores pre-shuffle dimensions directly."""
        return self.Hp, self.Wp

    def get_final_grid_dims(self) -> Tuple[int, int]:
        """Return post-shuffle grid dimensions."""
        return self.Gh, self.Gw

    def get_token_count(self) -> int:
        """DINOv3 uses 1 token per grid cell."""
        return self.Gh * self.Gw

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "dinov3",
            "HpWp": (self.Hp, self.Wp),
            "GhGw": (self.Gh, self.Gw),
        }

    @classmethod
    def from_raw(cls, raw_data: Any, pixel_shuffle_factor: int = 2) -> "DINOv3Grid":
        """Create from DINOv3 dict format."""
        if isinstance(raw_data, dict):
            if "GhGw" in raw_data and "HpWp" in raw_data:
                Hp, Wp = raw_data["HpWp"]
                Gh, Gw = raw_data["GhGw"]
                return cls(Hp=Hp, Wp=Wp, Gh=Gh, Gw=Gw)
            elif "GhGw" in raw_data:
                # Only have post-shuffle dims, compute pre-shuffle
                Gh, Gw = raw_data["GhGw"]
                Hp = Gh * pixel_shuffle_factor
                Wp = Gw * pixel_shuffle_factor
                return cls(Hp=Hp, Wp=Wp, Gh=Gh, Gw=Gw)
        raise ValueError(f"Cannot create DINOv3Grid from {type(raw_data)}: {raw_data}")


@dataclass
class SigLIPGrid(ImageGrid):
    """Grid representation for SigLIP with fixed-size image splitting."""

    rows: int  # Number of image rows (tiles)
    cols: int  # Number of image columns (tiles)
    tokens_per_tile: int  # Number of tokens per tile (mp_image_token_length)

    def get_modality_projector_dims(self, pixel_shuffle_factor: int) -> Tuple[int, int]:
        """For SigLIP, each tile produces a fixed number of patches.

        Each tile is processed independently by the vision encoder to produce
        patch embeddings. For a 224x224 tile with 16x16 patches, we get 14x14=196 patches.

        The modality projector needs the total patch grid dimensions across all tiles.
        For mp_image_token_length=49, we have sqrt(49)=7 tokens per side per tile.
        Working backwards: 7 * pixel_shuffle_factor = 14 patches per side per tile.
        """
        # For SigLIP, each tile is processed independently as a 224x224 image
        # This produces exactly 196 patches (14x14) per tile regardless of tokens_per_tile
        # The modality projector then reduces this to tokens_per_tile through pixel shuffle
        patches_per_tile_side = 14  # Fixed: 224x224 tile / 16x16 patches = 14x14 patches per tile
        return self.rows * patches_per_tile_side, self.cols * patches_per_tile_side

    def get_final_grid_dims(self) -> Tuple[int, int]:
        """Return tile grid dimensions."""
        return self.rows, self.cols

    def get_token_count(self) -> int:
        """SigLIP uses multiple tokens per tile."""
        return self.rows * self.cols * self.tokens_per_tile

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "siglip",
            "grid": (self.rows, self.cols),
            "tokens_per_tile": self.tokens_per_tile,
        }

    @classmethod
    def from_raw(
        cls, raw_data: Any, pixel_shuffle_factor: int = 2, tokens_per_tile: int = 64
    ) -> "SigLIPGrid":
        """Create from SigLIP tuple format."""
        if isinstance(raw_data, (tuple, list)):
            # Handle nested lists (e.g., [[12, 16]])
            while isinstance(raw_data, (list, tuple)) and len(raw_data) == 1:
                raw_data = raw_data[0]
            
            # If we still have a list/tuple with at least 2 elements
            if isinstance(raw_data, (list, tuple)) and len(raw_data) >= 2:
                # If first element is also a list/tuple, take it
                if isinstance(raw_data[0], (list, tuple)):
                    raw_data = raw_data[0]
                # Now ensure we have two integers for rows and cols
                if len(raw_data) >= 2:
                    rows, cols = int(raw_data[0]), int(raw_data[1])
                    return cls(rows=rows, cols=cols, tokens_per_tile=tokens_per_tile)
            
            # If we couldn't extract rows and cols, return default 1x1 grid
            return cls(rows=1, cols=1, tokens_per_tile=tokens_per_tile)
        elif isinstance(raw_data, dict) and "grid" in raw_data:
            grid = raw_data["grid"]
            tpt = raw_data.get("tokens_per_tile", tokens_per_tile)
            return cls(rows=grid[0], cols=grid[1], tokens_per_tile=tpt)
        raise ValueError(f"Cannot create SigLIPGrid from {type(raw_data)}: {raw_data}")


class GridFactory:
    """Factory for creating appropriate grid objects based on configuration."""

    @staticmethod
    def create_from_raw(raw_data: Any, config: Any) -> ImageGrid:
        """
        Create appropriate grid object based on raw data and config.

        Args:
            raw_data: Raw grid data (dict, tuple, etc.)
            config: VLMConfig object with architecture info

        Returns:
            ImageGrid subclass instance
        """
        if raw_data is None:
            # Default to minimal grid
            if config.vit_architecture == "dinov3":
                return DINOv3Grid(
                    Hp=config.mp_pixel_shuffle_factor,
                    Wp=config.mp_pixel_shuffle_factor,
                    Gh=1,
                    Gw=1,
                )
            else:
                return SigLIPGrid(
                    rows=1, cols=1, tokens_per_tile=config.mp_image_token_length
                )

        # First check architecture preference
        if config.vit_architecture == "dinov3":
            # DINOv3 mode - prefer DINOv3Grid
            if isinstance(raw_data, dict):
                if "GhGw" in raw_data or "HpWp" in raw_data:
                    return DINOv3Grid.from_raw(raw_data, config.mp_pixel_shuffle_factor)
            elif isinstance(raw_data, (tuple, list)) and len(raw_data) >= 1:
                # Handle nested lists (e.g., [[12, 16]])
                while len(raw_data) == 1 and isinstance(raw_data[0], (list, tuple)):
                    raw_data = raw_data[0]
                # Ensure we have at least 2 elements
                if len(raw_data) < 2:
                    return DINOv3Grid(
                        Hp=config.mp_pixel_shuffle_factor,
                        Wp=config.mp_pixel_shuffle_factor,
                        Gh=1,
                        Gw=1,
                    )
                # Convert tuple to DINOv3 format
                # Assume these are post-shuffle dims
                Gh, Gw = raw_data[0], raw_data[1]
                Hp = Gh * config.mp_pixel_shuffle_factor
                Wp = Gw * config.mp_pixel_shuffle_factor
                return DINOv3Grid(Hp=Hp, Wp=Wp, Gh=Gh, Gw=Gw)
        else:
            # SigLIP mode - prefer SigLIPGrid
            if isinstance(raw_data, (tuple, list)):
                return SigLIPGrid.from_raw(
                    raw_data,
                    config.mp_pixel_shuffle_factor,
                    config.mp_image_token_length,
                )
            elif isinstance(raw_data, dict) and "grid" in raw_data:
                return SigLIPGrid.from_raw(
                    raw_data,
                    config.mp_pixel_shuffle_factor,
                    config.mp_image_token_length,
                )

        # Fallback: try to infer from data format
        if isinstance(raw_data, dict):
            if "GhGw" in raw_data or "HpWp" in raw_data:
                return DINOv3Grid.from_raw(raw_data, config.mp_pixel_shuffle_factor)
            elif "grid" in raw_data:
                return SigLIPGrid.from_raw(
                    raw_data,
                    config.mp_pixel_shuffle_factor,
                    config.mp_image_token_length,
                )
        elif isinstance(raw_data, (tuple, list)):
            # Default to SigLIP for tuple format
            return SigLIPGrid.from_raw(
                raw_data, config.mp_pixel_shuffle_factor, config.mp_image_token_length
            )

        raise ValueError(f"Cannot create grid from {type(raw_data)}: {raw_data}")


def normalize_grids(grids: list, config: Any) -> list[ImageGrid]:
    """
    Convert a list of raw grid data to normalized ImageGrid objects.

    Args:
        grids: List of raw grid data (dicts, tuples, etc.)
        config: VLMConfig object

    Returns:
        List of ImageGrid objects
    """
    if not grids:
        return []

    normalized = []
    for grid in grids:
        try:
            normalized.append(GridFactory.create_from_raw(grid, config))
        except ValueError as e:
            # Make this a hard error instead of warning with fallback
            raise ValueError(f"Failed to normalize grid {grid}: {e}")

    return normalized
