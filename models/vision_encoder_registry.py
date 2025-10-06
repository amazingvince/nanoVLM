"""Registry for vision encoders in nanoVLM.

This module manages available vision encoders and their configurations,
providing a factory pattern for creating encoder instances.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Type

from models.vision_encoder_base import VisionEncoderBase


@dataclass
class VisionEncoderConfig:
    """Configuration for a vision encoder type.

    :param model_id: HuggingFace model ID or local path
    :param hidden_dim: Output hidden dimension
    :param patch_size: Patch size in pixels
    :param image_size: Default input image size
    :param has_cls: Whether encoder has CLS token
    :param num_register_tokens: Number of register tokens (e.g., DINOv3)
    :param preprocessing: Preprocessing configuration
    """
    model_id: str
    hidden_dim: int
    patch_size: int
    image_size: int
    has_cls: bool = False
    num_register_tokens: int = 0
    preprocessing: Dict = None

    def __post_init__(self):
        if self.preprocessing is None:
            self.preprocessing = {}


# Registry of available vision encoders and their configurations
VISION_ENCODER_CONFIGS = {
    "siglip": VisionEncoderConfig(
        model_id="google/siglip2-base-patch16-512",
        hidden_dim=768,
        patch_size=16,
        image_size=512,
        has_cls=False,
        preprocessing={
            "mean": None,  # SigLIP in nanoVLM doesn't use normalization
            "std": None,
            "interpolation": "bicubic",
        }
    ),
    "dinov3-small": VisionEncoderConfig(
        model_id="facebook/dinov3-vits16plus-pretrain-lvd1689m",  # 30M params - vits16plus
        hidden_dim=384,
        patch_size=16,
        image_size=518,  # DINOv3 uses 518x518 default
        has_cls=True,
        num_register_tokens=4,
        preprocessing={
            "mean": [0.485, 0.456, 0.406],  # ImageNet normalization
            "std": [0.229, 0.224, 0.225],
            "interpolation": "bicubic",
            "rescale_factor": 1.0 / 255.0,  # DINOv3 specific
        }
    ),
    "dinov3-base": VisionEncoderConfig(
        model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",  # Fixed: removed "plus"
        hidden_dim=768,
        patch_size=16,
        image_size=518,
        has_cls=True,
        num_register_tokens=4,
        preprocessing={
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "bicubic",
            "rescale_factor": 1.0 / 255.0,
        }
    ),
}

# Registry to store encoder classes
_ENCODER_REGISTRY: Dict[str, Type[VisionEncoderBase]] = {}


def register_encoder(name: str):
    """Decorator to register a vision encoder class.

    :param name: Name of the encoder type (e.g., "siglip", "dinov3")
    """
    def decorator(cls: Type[VisionEncoderBase]):
        _ENCODER_REGISTRY[name] = cls
        return cls
    return decorator


def get_encoder_class(encoder_type: str) -> Type[VisionEncoderBase]:
    """Get the encoder class for a given type.

    :param encoder_type: Type of encoder (e.g., "siglip", "dinov3-small")
    :return: Encoder class
    :raises ValueError: If encoder type not found
    """
    # Map specific variants to base type for class lookup
    base_type = encoder_type.split("-")[0] if "-" in encoder_type else encoder_type

    if base_type not in _ENCODER_REGISTRY:
        available = list(_ENCODER_REGISTRY.keys())
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. Available: {available}"
        )

    return _ENCODER_REGISTRY[base_type]


def get_encoder_config(encoder_type: str) -> VisionEncoderConfig:
    """Get configuration for a vision encoder type.

    :param encoder_type: Type of encoder
    :return: Encoder configuration
    :raises ValueError: If encoder type not found
    """
    if encoder_type not in VISION_ENCODER_CONFIGS:
        available = list(VISION_ENCODER_CONFIGS.keys())
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. Available: {available}"
        )

    return VISION_ENCODER_CONFIGS[encoder_type]


def create_vision_encoder(cfg, load_pretrained: bool = True) -> VisionEncoderBase:
    """Factory method to create a vision encoder instance.

    :param cfg: VLMConfig with vision_encoder_type field
    :param load_pretrained: Whether to load pretrained weights
    :return: Vision encoder instance
    """
    encoder_type = getattr(cfg, "vision_encoder_type", "siglip")
    encoder_config = get_encoder_config(encoder_type)

    # Update cfg with encoder-specific values
    cfg.vit_hidden_dim = encoder_config.hidden_dim
    cfg.vit_patch_size = encoder_config.patch_size
    cfg.vit_img_size = encoder_config.image_size
    cfg.vit_cls_flag = encoder_config.has_cls
    cfg.vit_num_register_tokens = encoder_config.num_register_tokens
    cfg.vit_model_type = encoder_config.model_id

    # Store encoder type and config for later use
    cfg._encoder_config = encoder_config
    cfg._encoder_type = encoder_type

    # Get encoder class and create instance
    encoder_class = get_encoder_class(encoder_type)

    if load_pretrained:
        encoder = encoder_class.from_pretrained(cfg)
    else:
        encoder = encoder_class(cfg)

    return encoder


def list_available_encoders() -> list:
    """List all available encoder types.

    :return: List of encoder type names
    """
    return list(VISION_ENCODER_CONFIGS.keys())