"""DINOv3 vision encoder wrapper for nanoVLM.

This module wraps the HuggingFace transformers DINOv3 implementation
to conform to the VisionEncoderBase interface.
"""

from typing import Dict

import torch
import torch.nn as nn

from models.vision_encoder_base import VisionEncoderBase, VisionEncoderOutput
from models.vision_encoder_registry import register_encoder


@register_encoder("dinov3")
class DINOv3Encoder(VisionEncoderBase):
    """DINOv3 vision encoder wrapper.

    This wraps the HuggingFace transformers DINOv3 model to provide a consistent
    interface with other vision encoders. DINOv3 features:
    - 2D RoPE position embeddings
    - Register tokens
    - LayerScale and DropPath
    - Optional SwiGLU FFN
    - CLS token for global representation
    """

    def __init__(self, cfg):
        """Initialize DINOv3 encoder.

        :param cfg: VLMConfig containing encoder configuration
        """
        super().__init__(cfg)

        # Import here to avoid dependency if not using DINOv3
        from transformers import AutoModel

        # Create DINOv3 model
        self.model = AutoModel.from_pretrained(
            cfg.vit_model_type,
            trust_remote_code=True,  # Required for DINOv3
        )

        # Auto-detect register tokens from model config
        model_config = self.model.config
        self._num_register_tokens = getattr(model_config, "num_register_tokens", 4)

        # Validate against user-provided value if present
        user_register_tokens = getattr(cfg, "vit_num_register_tokens", None)
        if user_register_tokens is not None and user_register_tokens != self._num_register_tokens:
            print(f"Warning: Config specifies {user_register_tokens} register tokens but model has {self._num_register_tokens}. Using model value.")

        # Store configuration
        self.hidden_dim = cfg.vit_hidden_dim
        self._patch_size = cfg.vit_patch_size
        self._image_size = cfg.vit_img_size
        self._has_cls = cfg.vit_cls_flag

        # DINOv3-specific parameters
        self._rope_theta = getattr(cfg, "vit_rope_theta", 100.0)
        self._max_resolution = getattr(cfg, "vit_max_resolution", 1024)
        self._training_resolution = getattr(model_config, "image_size", 224)  # DINOv3 training resolution

        print(f"DINOv3 initialized: register_tokens={self._num_register_tokens}, training_res={self._training_resolution}, max_res={self._max_resolution}")

    def forward(self, images: torch.Tensor) -> VisionEncoderOutput:
        """Encode images using DINOv3.

        :param images: Input images [batch_size, 3, height, width]
        :return: VisionEncoderOutput with features and pooled output
        """
        # Check image size for RoPE extrapolation warning
        _, _, h, w = images.shape
        if max(h, w) > self._max_resolution:
            print(f"Warning: Image size {h}x{w} exceeds max resolution {self._max_resolution}. "
                  f"DINOv3's 2D RoPE may have degraded performance beyond training resolution ({self._training_resolution}).")

        # Get outputs from DINOv3
        outputs = self.model(images, return_dict=True)

        # Extract features - DINOv3 returns last_hidden_state
        all_features = outputs.last_hidden_state  # [B, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = all_features.shape

        # DINOv3 token layout: [CLS, register_tokens..., patch_tokens...]
        # We need to extract just the patch tokens for the VLM
        num_prefix_tokens = 1 + self._num_register_tokens  # CLS + registers

        # Validate token layout
        expected_patches = (h // self._patch_size) * (w // self._patch_size)
        expected_seq_len = num_prefix_tokens + expected_patches
        if seq_len != expected_seq_len:
            raise ValueError(
                f"Token layout mismatch! Expected {expected_seq_len} tokens "
                f"(1 CLS + {self._num_register_tokens} registers + {expected_patches} patches), "
                f"but got {seq_len}. Check register token count and image dimensions."
            )

        # Extract patch features (excluding CLS and register tokens)
        patch_features = all_features[:, num_prefix_tokens:, :]

        # Extract CLS token as pooled output
        pooled_output = all_features[:, 0, :]  # CLS token

        # Calculate grid shape based on number of patches
        batch_size, num_patches, _ = patch_features.shape
        grid_size = int(num_patches ** 0.5)
        grid_shape = (grid_size, grid_size)

        return VisionEncoderOutput(
            features=patch_features,
            pooled_output=pooled_output,
            num_patches=num_patches,
            grid_shape=grid_shape,
        )

    @property
    def output_dim(self) -> int:
        """Get output dimension.

        :return: Hidden dimension (384 for small, 768 for base)
        """
        return self.hidden_dim

    @property
    def num_patches(self) -> int:
        """Get number of patches for standard input.

        :return: Number of patches
        """
        return (self._image_size // self._patch_size) ** 2

    @property
    def patch_size(self) -> int:
        """Get patch size.

        :return: Patch size in pixels (16)
        """
        return self._patch_size

    @property
    def has_cls_token(self) -> bool:
        """Check if has CLS token.

        :return: True (DINOv3 uses CLS token)
        """
        return True

    @property
    def has_register_tokens(self) -> bool:
        """Check if has register tokens.

        :return: True (DINOv3 uses register tokens)
        """
        return True

    @property
    def num_register_tokens(self) -> int:
        """Get number of register tokens.

        :return: Number of register tokens (4 by default)
        """
        return self._num_register_tokens

    def get_preprocessing_config(self) -> Dict:
        """Get preprocessing configuration.

        :return: Preprocessing parameters for DINOv3
        """
        return {
            "image_size": self._image_size,
            "patch_size": self._patch_size,
            "mean": [0.485, 0.456, 0.406],  # ImageNet normalization
            "std": [0.229, 0.224, 0.225],
            "interpolation": "bicubic",
            "rescale_factor": 1.0 / 255.0,  # DINOv3 specific
            "do_rescale": True,
            "do_normalize": True,
        }

    @classmethod
    def from_pretrained(cls, cfg) -> 'DINOv3Encoder':
        """Load pretrained DINOv3 weights.

        :param cfg: VLMConfig with model specification
        :return: DINOv3Encoder with loaded weights
        """
        # Simply create a new instance - __init__ handles loading
        return cls(cfg)

    def freeze(self) -> None:
        """Freeze all encoder parameters (recommended for DINOv3 in VLM)."""
        super().freeze()
        # Also set model to eval mode for DINOv3
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0  # Disable dropout when frozen