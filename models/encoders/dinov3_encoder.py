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
        from transformers import AutoModel, AutoImageProcessor

        # Prepare config overrides for DINOv3-specific features
        config_overrides = {
            "trust_remote_code": True,  # Required for DINOv3
        }

        # Pass DINOv3-specific parameters if they are set
        if hasattr(cfg, "vit_layerscale_value") and cfg.vit_layerscale_value is not None:
            config_overrides["layerscale_value"] = cfg.vit_layerscale_value
        if hasattr(cfg, "vit_drop_path_rate") and cfg.vit_drop_path_rate is not None:
            config_overrides["drop_path_rate"] = cfg.vit_drop_path_rate
        if hasattr(cfg, "vit_rope_theta") and cfg.vit_rope_theta is not None:
            config_overrides["rope_theta"] = cfg.vit_rope_theta
        if hasattr(cfg, "vit_pos_embed_shift") and cfg.vit_pos_embed_shift is not None:
            config_overrides["pos_embed_shift"] = cfg.vit_pos_embed_shift
        if hasattr(cfg, "vit_pos_embed_jitter") and cfg.vit_pos_embed_jitter is not None:
            config_overrides["pos_embed_jitter"] = cfg.vit_pos_embed_jitter
        if hasattr(cfg, "vit_pos_embed_rescale") and cfg.vit_pos_embed_rescale is not None:
            config_overrides["pos_embed_rescale"] = cfg.vit_pos_embed_rescale

        # Create DINOv3 model with config overrides
        self.model = AutoModel.from_pretrained(
            cfg.vit_model_type,
            **config_overrides
        )

        # Create the official processor for proper preprocessing
        self.processor = AutoImageProcessor.from_pretrained(
            cfg.vit_model_type,
            trust_remote_code=True
        )

        # Auto-detect register tokens from model config
        model_config = self.model.config
        self._num_register_tokens = getattr(model_config, "num_register_tokens", 0)

        # Warn if model doesn't have register tokens configured
        if not hasattr(model_config, "num_register_tokens"):
            print(f"Warning: Model config doesn't specify num_register_tokens. Assuming {self._num_register_tokens}.")

        # Validate that our config overrides were applied
        if hasattr(cfg, "vit_layerscale_value") and cfg.vit_layerscale_value is not None:
            actual_value = getattr(model_config, "layerscale_value", None)
            if actual_value != cfg.vit_layerscale_value:
                print(f"Warning: LayerScale override may not have been applied. Expected {cfg.vit_layerscale_value}, got {actual_value}")

        if hasattr(cfg, "vit_drop_path_rate") and cfg.vit_drop_path_rate is not None:
            actual_value = getattr(model_config, "drop_path_rate", None)
            if actual_value != cfg.vit_drop_path_rate:
                print(f"Warning: DropPath override may not have been applied. Expected {cfg.vit_drop_path_rate}, got {actual_value}")

        # Store configuration
        self.hidden_dim = cfg.vit_hidden_dim
        self._patch_size = cfg.vit_patch_size
        self._image_size = cfg.vit_img_size
        self._has_cls = cfg.vit_cls_flag

        # Store DINOv3-specific parameters from model config
        self._rope_theta = getattr(model_config, "rope_theta", 100.0)
        self._max_resolution = getattr(cfg, "vit_max_resolution", 1024)
        self._training_resolution = getattr(model_config, "image_size", 224)
        self._layerscale_value = getattr(model_config, "layerscale_value", 1.0)
        self._drop_path_rate = getattr(model_config, "drop_path_rate", 0.0)
        self._pos_embed_shift = getattr(model_config, "pos_embed_shift", None)
        self._pos_embed_jitter = getattr(model_config, "pos_embed_jitter", None)
        self._pos_embed_rescale = getattr(model_config, "pos_embed_rescale", 2.0)

        # Print configuration summary
        print(f"DINOv3 initialized with:")
        print(f"  Model: {cfg.vit_model_type}")
        print(f"  Register tokens: {self._num_register_tokens}")
        print(f"  Training resolution: {self._training_resolution}")
        print(f"  Max resolution: {self._max_resolution}")
        print(f"  LayerScale: {self._layerscale_value}")
        print(f"  DropPath: {self._drop_path_rate}")
        if self._pos_embed_shift or self._pos_embed_jitter or self._pos_embed_rescale != 2.0:
            print(f"  Position augmentation: shift={self._pos_embed_shift}, jitter={self._pos_embed_jitter}, rescale={self._pos_embed_rescale}")

    def preprocess_with_processor(self, images):
        """Preprocess images using the official DINOv3 processor.

        :param images: PIL images or tensors
        :return: Preprocessed tensor ready for model
        """
        # If images are already tensors, we can use them directly
        # Otherwise, use the processor
        if not isinstance(images, torch.Tensor):
            # Use the official processor for PIL images
            inputs = self.processor(images=images, return_tensors="pt")
            return inputs["pixel_values"]
        return images

    def train(self, mode: bool = True):
        """Set training mode for both encoder and internal model.

        :param mode: Whether to set training mode (True) or eval mode (False)
        :return: Self
        """
        super().train(mode)
        self.model.train(mode)
        # Log when position augmentations will be active
        if mode and (self._pos_embed_shift or self._pos_embed_jitter or self._pos_embed_rescale != 2.0):
            print(f"DINOv3 training mode: Position augmentations active (shift={self._pos_embed_shift}, jitter={self._pos_embed_jitter}, rescale={self._pos_embed_rescale})")
        return self

    def eval(self):
        """Set eval mode for both encoder and internal model.

        :return: Self
        """
        return self.train(False)

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
        """Get preprocessing configuration from the official processor.

        :return: Preprocessing parameters for DINOv3
        """
        # Use the official processor's configuration
        proc_config = {
            "image_size": self._image_size,
            "patch_size": self._patch_size,
            "mean": getattr(self.processor, "image_mean", [0.485, 0.456, 0.406]),
            "std": getattr(self.processor, "image_std", [0.229, 0.224, 0.225]),
            "interpolation": "bilinear",  # DINOv3 uses BILINEAR
            "rescale_factor": getattr(self.processor, "rescale_factor", 1.0 / 255.0),
            "do_rescale": getattr(self.processor, "do_rescale", True),
            "do_normalize": getattr(self.processor, "do_normalize", True),
            # DINOv3 preprocessing order: rescale → resize → normalize
            "preprocessing_order": ["rescale", "resize", "normalize"],
            "processor": self.processor,  # Include processor for direct use
        }
        return proc_config

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
        # Set model to eval mode to disable position augmentations and dropout
        self.eval()
        # Ensure dropout is disabled
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0  # Disable dropout when frozen
        print("DINOv3 frozen: Position augmentations and dropout disabled")