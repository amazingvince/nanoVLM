"""
Encoder-specific adapters for handling different vision architectures.

This module provides a clean abstraction for handling the fundamental differences
between vision encoders like SigLIP (tile-based) and DINOv3 (dynamic resolution).
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod

from models.grid_abstraction import ImageGrid, SigLIPGrid, DINOv3Grid, GridFactory


class VisionEncoderAdapter(ABC):
    """Abstract base class for vision encoder adapters."""
    
    @abstractmethod
    def prepare_images(self, images: List[torch.Tensor], grids: List[Any]) -> Tuple[torch.Tensor, List[ImageGrid]]:
        """
        Prepare images for the vision encoder.
        
        Args:
            images: List of image tensors (potentially with different numbers of tiles)
            grids: List of grid information
            
        Returns:
            Batched tensor ready for vision encoder and normalized grids
        """
        pass
    
    @abstractmethod
    def process_embeddings(
        self, 
        embeddings: torch.Tensor, 
        grids: List[ImageGrid],
        modality_projector: nn.Module
    ) -> List[torch.Tensor]:
        """
        Process vision encoder embeddings through modality projector.
        
        Args:
            embeddings: Raw embeddings from vision encoder
            grids: Grid information for each image
            modality_projector: The modality projector module
            
        Returns:
            List of projected embeddings for each image
        """
        pass
    
    @abstractmethod
    def get_expected_tokens(self, grids: List[ImageGrid]) -> int:
        """Calculate expected number of tokens after modality projection."""
        pass


class SigLIPAdapter(VisionEncoderAdapter):
    """Adapter for SigLIP's tile-based processing."""
    
    def __init__(self, config):
        self.config = config
        # Calculate patches per tile based on config
        self.patches_per_tile = (config.vit_img_size // config.vit_patch_size) ** 2
        
    def prepare_images(self, images: List[torch.Tensor], grids: List[Any]) -> Tuple[torch.Tensor, List[ImageGrid]]:
        """
        For SigLIP, images are already split into tiles by SplitImage transform.
        We need to track which tiles belong to which image.
        
        Note: grids is a list where each element contains grids for all images in that batch item.
        images is a flattened list of all tiles from all images.
        """
        # Flatten the grid info if it's nested (batch-wise structure)
        flattened_grids = []
        for batch_grids in grids:
            if isinstance(batch_grids, list):
                flattened_grids.extend(batch_grids)
            else:
                flattened_grids.append(batch_grids)
        
        # Normalize grids
        normalized_grids = []
        tile_counts = []
        
        for grid_raw in flattened_grids:
            # For SigLIP, grid_raw should be a tuple (n_h, n_w) from SplitImage
            if isinstance(grid_raw, (tuple, list)) and len(grid_raw) >= 2:
                n_h, n_w = int(grid_raw[0]), int(grid_raw[1])
            else:
                n_h, n_w = 1, 1
            
            grid = SigLIPGrid(rows=n_h, cols=n_w, tokens_per_tile=self.config.mp_image_token_length)
            normalized_grids.append(grid)
            tile_counts.append(grid.rows * grid.cols)
        
        # Concatenate all tiles into a single batch
        all_tiles = []
        for img_tiles in images:
            if img_tiles.dim() == 3:
                img_tiles = img_tiles.unsqueeze(0)
            all_tiles.append(img_tiles)
        
        batched_tiles = torch.cat(all_tiles, dim=0) if all_tiles else torch.empty(0, 3, 224, 224)
        
        # Store tile mapping for later use
        self.tile_to_image_mapping = []
        for img_idx, count in enumerate(tile_counts):
            self.tile_to_image_mapping.extend([img_idx] * count)
        
        return batched_tiles, normalized_grids
    
    def process_embeddings(
        self, 
        embeddings: torch.Tensor, 
        grids: List[ImageGrid],
        modality_projector: nn.Module
    ) -> List[torch.Tensor]:
        """
        Process SigLIP embeddings by grouping tiles back to their original images.
        """
        projected = []
        tile_idx = 0
        
        for grid in grids:
            num_tiles = grid.rows * grid.cols
            
            # Extract embeddings for this image's tiles
            tile_embeddings = embeddings[tile_idx:tile_idx + num_tiles]
            
            # Apply modality projector
            # For SigLIP, each tile produces a fixed number of tokens
            proj_embd = modality_projector(tile_embeddings)
            
            # Reshape to combine all tiles' tokens for this image
            # Shape: [num_tiles, tokens_per_tile, hidden_dim] -> [1, total_tokens, hidden_dim]
            total_tokens = num_tiles * self.config.mp_image_token_length
            proj_embd = proj_embd.reshape(1, total_tokens, -1)
            
            projected.append(proj_embd)
            tile_idx += num_tiles
        
        return projected
    
    def get_expected_tokens(self, grids: List[ImageGrid]) -> int:
        """Calculate total expected tokens for all images."""
        return sum(grid.get_token_count() for grid in grids)


class DINOv3Adapter(VisionEncoderAdapter):
    """Adapter for DINOv3's dynamic resolution processing."""
    
    def __init__(self, config):
        self.config = config
        
    def prepare_images(self, images: List[torch.Tensor], grids: List[Any]) -> Tuple[torch.Tensor, List[ImageGrid]]:
        """
        For DINOv3, images are aspect-preserved and may have different sizes.
        We pad them to the same size for batching.
        """
        import torch.nn.functional as F
        
        # Normalize grids
        normalized_grids = []
        for grid_raw in grids:
            grid = GridFactory.create_from_raw(grid_raw, self.config)
            if not isinstance(grid, DINOv3Grid):
                # Convert to DINOv3 grid if needed
                if isinstance(grid_raw, dict) and "HpWp" in grid_raw:
                    grid = DINOv3Grid.from_raw(grid_raw, self.config.mp_pixel_shuffle_factor)
                else:
                    # Fallback
                    grid = DINOv3Grid(Hp=14, Wp=14, Gh=7, Gw=7)
            normalized_grids.append(grid)
        
        # Handle list of images
        if not images:
            return torch.empty(0, 3, 224, 224), normalized_grids
        
        # Ensure all images are 4D tensors
        processed_images = []
        for img in images:
            if img.dim() == 3:
                img = img.unsqueeze(0)
            processed_images.append(img)
        
        # Find max dimensions for padding
        max_h = max(img.shape[2] for img in processed_images)
        max_w = max(img.shape[3] for img in processed_images)
        
        # Store original sizes for proper grid handling
        self.original_sizes = [(img.shape[2], img.shape[3]) for img in processed_images]
        
        # Pad images to same size
        padded_images = []
        for img in processed_images:
            pad_h = max_h - img.shape[2]
            pad_w = max_w - img.shape[3]
            # Pad on right and bottom
            padded = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
            padded_images.append(padded)
        
        # Batch padded images
        batched = torch.cat(padded_images, dim=0)
        
        return batched, normalized_grids
    
    def process_embeddings(
        self, 
        embeddings: torch.Tensor, 
        grids: List[ImageGrid],
        modality_projector: nn.Module
    ) -> List[torch.Tensor]:
        """
        Process DINOv3 embeddings with dynamic grid dimensions.
        
        Note: The modality projector will handle removing special tokens (CLS + registers)
        if configured to do so via mp_handle_special_tokens.
        """
        projected = []
        
        for i, grid in enumerate(grids):
            # Get pre-shuffle dimensions for modality projector
            # These are the patch grid dimensions (Hp, Wp) before pixel shuffle
            Hp, Wp = grid.get_modality_projector_dims(self.config.mp_pixel_shuffle_factor)
            
            # Extract embeddings for this image
            img_embeddings = embeddings[i:i+1]
            
            # Apply modality projector with 2D pixel shuffle
            # The modality projector will handle special tokens and pixel shuffle
            proj_embd = modality_projector(img_embeddings, gh=Hp, gw=Wp)
            
            projected.append(proj_embd)
        
        return projected
    
    def get_expected_tokens(self, grids: List[ImageGrid]) -> int:
        """Calculate total expected tokens for all images."""
        return sum(grid.get_token_count() for grid in grids)


def create_encoder_adapter(config) -> VisionEncoderAdapter:
    """Factory function to create the appropriate encoder adapter."""
    if config.vit_architecture == "dinov3":
        return DINOv3Adapter(config)
    else:
        return SigLIPAdapter(config)