"""Base abstraction for vision encoders in nanoVLM.

This module provides the interface that all vision encoders must implement,
enabling easy swapping between different vision backbones (SigLIP, DINOv3, CLIP, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class VisionEncoderOutput:
    """Output from a vision encoder.

    :param features: Main output features [batch_size, num_patches, hidden_dim]
    :param pooled_output: Optional pooled/CLS token output [batch_size, hidden_dim]
    :param num_patches: Number of patch tokens (excluding CLS/register tokens)
    :param grid_shape: Optional (height, width) of patch grid
    """
    features: torch.Tensor
    pooled_output: Optional[torch.Tensor] = None
    num_patches: Optional[int] = None
    grid_shape: Optional[Tuple[int, int]] = None


class VisionEncoderBase(nn.Module):
    """Abstract base class for vision encoders.

    All vision encoders (SigLIP, DINOv3, CLIP, etc.) should inherit from this class
    and implement the required methods.
    """

    def __init__(self, cfg):
        """Initialize vision encoder.

        :param cfg: VLMConfig containing encoder configuration
        """
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, images: torch.Tensor) -> VisionEncoderOutput:
        """Encode images to feature representations.

        :param images: Input images [batch_size, 3, height, width]
        :return: VisionEncoderOutput containing features and metadata
        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Get the output dimension of encoded features.

        :return: Hidden dimension of output features
        """
        pass

    @property
    @abstractmethod
    def num_patches(self) -> int:
        """Get the number of output patches for standard input size.

        :return: Number of patch tokens (excluding special tokens)
        """
        pass

    @property
    @abstractmethod
    def patch_size(self) -> int:
        """Get the patch size used by the encoder.

        :return: Size of patches in pixels
        """
        pass

    @property
    @abstractmethod
    def has_cls_token(self) -> bool:
        """Check if encoder uses a CLS token.

        :return: True if encoder has CLS token
        """
        pass

    @property
    def has_register_tokens(self) -> bool:
        """Check if encoder uses register tokens (e.g., DINOv3).

        :return: True if encoder has register tokens
        """
        return False

    @property
    def num_register_tokens(self) -> int:
        """Get number of register tokens if applicable.

        :return: Number of register tokens
        """
        return 0

    @abstractmethod
    def get_preprocessing_config(self) -> Dict:
        """Get preprocessing configuration for this encoder.

        :return: Dictionary with preprocessing parameters
                 (e.g., image_size, mean, std, interpolation, etc.)
        """
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, cfg) -> 'VisionEncoderBase':
        """Load pretrained encoder weights.

        :param cfg: VLMConfig with encoder specification
        :return: Encoder instance with loaded weights
        """
        pass

    def freeze(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters.

        :return: Number of parameters with requires_grad=True
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Get total number of parameters.

        :return: Total parameter count
        """
        return sum(p.numel() for p in self.parameters())