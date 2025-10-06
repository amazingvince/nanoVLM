"""SigLIP vision encoder wrapper for nanoVLM.

This module wraps the existing ViT implementation to conform to the
VisionEncoderBase interface.
"""

from typing import Dict

import torch

from models.vision_encoder_base import VisionEncoderBase, VisionEncoderOutput
from models.vision_encoder_registry import register_encoder
from models.vision_transformer import ViT


@register_encoder("siglip")
class SigLIPEncoder(VisionEncoderBase):
    """SigLIP vision encoder wrapper.

    This wraps the existing ViT implementation to provide a consistent
    interface with other vision encoders.
    """

    def __init__(self, cfg):
        """Initialize SigLIP encoder.

        :param cfg: VLMConfig containing encoder configuration
        """
        super().__init__(cfg)
        self.vit = ViT(cfg)

    def forward(self, images: torch.Tensor) -> VisionEncoderOutput:
        """Encode images using SigLIP ViT.

        :param images: Input images [batch_size, 3, height, width]
        :return: VisionEncoderOutput with features
        """
        # Get features from ViT
        features = self.vit(images)

        # Calculate grid shape based on input size
        batch_size = images.shape[0]
        height, width = images.shape[-2:]
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        num_patches = grid_h * grid_w

        return VisionEncoderOutput(
            features=features,
            pooled_output=None,  # SigLIP doesn't use CLS token
            num_patches=num_patches,
            grid_shape=(grid_h, grid_w),
        )

    @property
    def output_dim(self) -> int:
        """Get output dimension.

        :return: Hidden dimension (768 for base model)
        """
        return self.cfg.vit_hidden_dim

    @property
    def num_patches(self) -> int:
        """Get number of patches for standard input.

        :return: Number of patches (32x32=1024 for 512px input)
        """
        return (self.cfg.vit_img_size // self.cfg.vit_patch_size) ** 2

    @property
    def patch_size(self) -> int:
        """Get patch size.

        :return: Patch size in pixels (16)
        """
        return self.cfg.vit_patch_size

    @property
    def has_cls_token(self) -> bool:
        """Check if has CLS token.

        :return: False (SigLIP doesn't use CLS token)
        """
        return False

    def get_preprocessing_config(self) -> Dict:
        """Get preprocessing configuration.

        :return: Preprocessing parameters for SigLIP
        """
        return {
            "image_size": self.cfg.vit_img_size,
            "patch_size": self.cfg.vit_patch_size,
            "mean": None,  # nanoVLM SigLIP doesn't use normalization
            "std": None,
            "interpolation": "bicubic",
        }

    @classmethod
    def from_pretrained(cls, cfg) -> 'SigLIPEncoder':
        """Load pretrained SigLIP weights.

        :param cfg: VLMConfig with model specification
        :return: SigLIPEncoder with loaded weights
        """
        encoder = cls.__new__(cls)
        VisionEncoderBase.__init__(encoder, cfg)
        encoder.vit = ViT.from_pretrained(cfg)
        return encoder