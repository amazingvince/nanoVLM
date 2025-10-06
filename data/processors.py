from typing import Dict, List, Optional, Tuple

import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, PreTrainedTokenizer

from data.custom_transforms import DynamicResize, GlobalAndSplitImages

TOKENIZERS_CACHE: Dict[str, PreTrainedTokenizer] = {}


def get_tokenizer(
    name: str,
    extra_special_tokens: Optional[Dict[str, str]] = None,
    chat_template: Optional[str] = None,
) -> PreTrainedTokenizer:
    """Get or create a cached tokenizer with optional special tokens and chat template.

    :param name: Model name or path for tokenizer
    :param extra_special_tokens: Dictionary of extra special tokens to add
    :param chat_template: Custom chat template for tokenizer
    :return: Configured tokenizer instance
    """
    if name not in TOKENIZERS_CACHE:
        tokenizer_init_kwargs = {"use_fast": True}
        if extra_special_tokens is not None:
            tokenizer_init_kwargs["extra_special_tokens"] = extra_special_tokens
        if chat_template is not None:
            tokenizer_init_kwargs["chat_template"] = chat_template
        tokenizer = AutoTokenizer.from_pretrained(
            name,
            **tokenizer_init_kwargs,
        )
        tokenizer.pad_token = tokenizer.eos_token
        TOKENIZERS_CACHE[name] = tokenizer
    return TOKENIZERS_CACHE[name]


def get_image_processor(
    max_img_size: int,
    splitted_image_size: int,
    resize_to_max_side_len: bool = False,
    encoder_type: str = "siglip",
    processor=None,  # Optional AutoImageProcessor for DINOv3
    compression_factor: int = 4,  # Modality projector compression factor
    mp_image_token_length: int = 64,  # Target tokens per window
) -> transforms.Compose:
    """Create image preprocessing pipeline with dynamic resizing and splitting.

    :param max_img_size: Maximum image size in pixels
    :param splitted_image_size: Size of split image patches
    :param resize_to_max_side_len: Whether to resize to max side length
    :param encoder_type: Type of vision encoder for preprocessing
    :param processor: Optional AutoImageProcessor for DINOv3 to ensure accurate preprocessing
    :param compression_factor: Modality projector compression factor (typically 4)
    :param mp_image_token_length: Target tokens per window after projection (typically 64)
    :return: Composed image transformation pipeline
    """
    transform_list = []

    # Determine if we need windowing for high-resolution support
    # DINOv3 at high res needs fixed windows, SigLIP can handle variable patches
    use_windowing = encoder_type.startswith("dinov3") and splitted_image_size > 256

    # Calculate window size for encoders that need it
    # Window must produce exactly mp_image_token_length tokens after compression
    # For patch16 with 4x compression: 256x256 window → 16x16 patches → 256 patches → 64 tokens
    patch_size = 16  # All our models use patch16
    if use_windowing:
        patches_per_window = mp_image_token_length * (compression_factor**2)
        patches_per_side = int(patches_per_window**0.5)
        window_size = patches_per_side * patch_size  # Should be 256 for default values
    else:
        window_size = splitted_image_size  # Not used in standard mode

    # DINOv3 requires specific preprocessing order and BILINEAR interpolation
    if encoder_type.startswith("dinov3"):
        # Use processor's config if available for accurate preprocessing
        if processor is not None:
            mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
            std = getattr(processor, "image_std", [0.229, 0.224, 0.225])
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        # For DINOv3: rescale → resize → normalize (order matters!)
        # Use BILINEAR interpolation as per DINOv3 reference
        transform_list.extend(
            [
                DynamicResize(
                    splitted_image_size,
                    max_img_size,
                    resize_to_max_side_len,
                    interpolation=InterpolationMode.BILINEAR,  # DINOv3 uses BILINEAR
                ),
                transforms.ToTensor(),  # Implicitly rescales [0,255] → [0,1]
                # ImageNet normalization for DINOv3 using processor's values
                transforms.Normalize(mean=mean, std=std),
                GlobalAndSplitImages(
                    patch_size,  # Pass actual patch size (16), not image size
                    use_windowing=use_windowing,
                    window_size=window_size,
                ),
            ]
        )
    else:
        # SigLIP and others: resize → tensor (no normalization)
        # Keep BICUBIC interpolation for SigLIP (default)
        transform_list.extend(
            [
                DynamicResize(
                    splitted_image_size, max_img_size, resize_to_max_side_len
                ),
                transforms.ToTensor(),
                GlobalAndSplitImages(
                    splitted_image_size,
                    use_windowing=False,  # SigLIP uses standard mode
                    window_size=window_size,  # Not used but passed for consistency
                ),
            ]
        )

    return transforms.Compose(transform_list)


def get_image_string(
    tokenizer: PreTrainedTokenizer,
    splitted_image_counts: List[Tuple[int, int]],
    mp_image_token_length: int,
) -> str:
    """Generate tokenized string representation for windowed images.

    With window-based splitting, each window (including global) produces exactly
    mp_image_token_length tokens.

    :param tokenizer: Tokenizer with image special tokens
    :param splitted_image_counts: List of (n_windows_h, n_windows_w) tuples for window counts
    :param mp_image_token_length: Number of image tokens per window
    :return: String with image tokens for all windows
    """
    image_string = ""

    for idx, (n_h, n_w) in enumerate(splitted_image_counts):
        if len(splitted_image_counts) > 1:
            image_string += f"<image: {idx}>"

        # For window-based approach:
        # - If (1,1): single window, no global needed
        # - If (n_h, n_w) with n_h*n_w > 1: we have 1 global + n_h*n_w local windows

        if n_h == 1 and n_w == 1:
            # Single window - just emit tokens for it
            image_string += tokenizer.image_token * mp_image_token_length
        else:
            # Multiple windows: global + local windows
            # Global window tokens
            if hasattr(tokenizer, "global_image_token"):
                image_string += tokenizer.global_image_token
            image_string += tokenizer.image_token * mp_image_token_length

            # Local window tokens with position indicators
            for i in range(n_h):
                for j in range(n_w):
                    # Add position token if available
                    pos_token_name = f"r{i + 1}c{j + 1}"
                    if hasattr(tokenizer, pos_token_name):
                        image_string += getattr(tokenizer, pos_token_name)
                    image_string += tokenizer.image_token * mp_image_token_length

    return image_string
