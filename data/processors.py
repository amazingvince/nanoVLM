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
) -> transforms.Compose:
    """Create image preprocessing pipeline with dynamic resizing and splitting.

    :param max_img_size: Maximum image size in pixels
    :param splitted_image_size: Size of split image patches
    :param resize_to_max_side_len: Whether to resize to max side length
    :param encoder_type: Type of vision encoder for preprocessing
    :param processor: Optional AutoImageProcessor for DINOv3 to ensure accurate preprocessing
    :return: Composed image transformation pipeline
    """
    transform_list = []

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
        transform_list.extend([
            DynamicResize(
                splitted_image_size,
                max_img_size,
                resize_to_max_side_len,
                interpolation=InterpolationMode.BILINEAR  # DINOv3 uses BILINEAR
            ),
            transforms.ToTensor(),  # Implicitly rescales [0,255] → [0,1]
            # ImageNet normalization for DINOv3 using processor's values
            transforms.Normalize(mean=mean, std=std),
            GlobalAndSplitImages(splitted_image_size),
        ])
    else:
        # SigLIP and others: resize → tensor (no normalization)
        # Keep BICUBIC interpolation for SigLIP (default)
        transform_list.extend([
            DynamicResize(splitted_image_size, max_img_size, resize_to_max_side_len),
            transforms.ToTensor(),
            GlobalAndSplitImages(splitted_image_size),
        ])

    return transforms.Compose(transform_list)


def get_image_string(
    tokenizer: PreTrainedTokenizer,
    splitted_image_counts: List[Tuple[int, int]],
    mp_image_token_length: int,
) -> str:
    """Generate tokenized string representation for split images with position tokens.
    
    :param tokenizer: Tokenizer with image special tokens
    :param splitted_image_counts: List of (height, width) tuples for split counts
    :param mp_image_token_length: Number of image tokens per patch
    :return: String with image tokens and position markers
    """
    image_string = ""
    # splitted_image_counts is a list of tuples (n_h, n_w)
    for idx, (n_h, n_w) in enumerate(splitted_image_counts):
        if len(splitted_image_counts) > 1:
            image_string += f"<image: {idx}>"
        if hasattr(tokenizer, "global_image_token"):
            image_string += tokenizer.global_image_token
            image_string += tokenizer.image_token * mp_image_token_length
            if (
                n_h == 1 and n_w == 1
            ):  # If there is only one patch, treat it as the global image
                continue
        for i in range(n_h):
            for j in range(n_w):
                image_string += getattr(tokenizer, f"r{i + 1}c{j + 1}")
                image_string += tokenizer.image_token * mp_image_token_length
    return image_string
