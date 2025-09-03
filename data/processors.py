import torchvision.transforms as transforms
from transformers import AutoTokenizer

from data.custom_transforms import DynamicResize, SplitImage

TOKENIZERS_CACHE = {}


def get_tokenizer(name, extra_special_tokens=None, chat_template=None):
    """Get tokenizer with support for both standard and Gemma tokenizers"""
    if name not in TOKENIZERS_CACHE:
        # Detect tokenizer type
        is_gemma = "gemma" in name.lower()

        # Prepare special tokens list
        if extra_special_tokens is not None:
            if isinstance(extra_special_tokens, dict):
                special_tokens_list = list(extra_special_tokens.values())
            else:
                special_tokens_list = list(extra_special_tokens)
        else:
            special_tokens_list = None

        if is_gemma:
            # Use AutoTokenizer which will automatically select GemmaTokenizer
            tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)

            # Add special tokens for Gemma if provided
            if special_tokens_list is not None:
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": special_tokens_list}
                )

            # Apply chat template if provided
            if chat_template is not None:
                tokenizer.chat_template = chat_template
        else:
            # Standard tokenizer loading (for Llama/SmolLM)
            tokenizer_init_kwargs = {"use_fast": True}
            if special_tokens_list is not None:
                tokenizer_init_kwargs["additional_special_tokens"] = special_tokens_list
            if chat_template is not None:
                tokenizer_init_kwargs["chat_template"] = chat_template
            tokenizer = AutoTokenizer.from_pretrained(name, **tokenizer_init_kwargs)

        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Set special token attributes if extra_special_tokens is a dict
        if extra_special_tokens is not None and isinstance(extra_special_tokens, dict):
            for key, value in extra_special_tokens.items():
                # Set both the token string and its ID as attributes
                setattr(tokenizer, key, value)
                # Get the token ID
                token_id = tokenizer.convert_tokens_to_ids(value)
                setattr(tokenizer, f"{key}_id", token_id)

        TOKENIZERS_CACHE[name] = tokenizer
    return TOKENIZERS_CACHE[name]


def get_image_processor(max_img_size, splitted_image_size, single_image_mode=False):
    """
    Create image processor.
    
    Args:
        max_img_size: Maximum size for image dimension
        splitted_image_size: Target size for each image/patch
        single_image_mode: If True, resize to splitted_image_size instead of splitting (for DINOv3)
    """
    if single_image_mode:
        # For DINOv3: resize entire image to 224x224
        return transforms.Compose([
            transforms.Resize((splitted_image_size, splitted_image_size)),
            transforms.ToTensor(),
            lambda x: (x, (1, 1))  # Return tensor and grid count (1x1)
        ])
    else:
        # For SigLIP: split into patches
        return transforms.Compose([
            DynamicResize(splitted_image_size, max_img_size),
            transforms.ToTensor(),
            SplitImage(splitted_image_size),
        ])


def get_image_string(tokenizer, splitted_image_counts, mp_image_token_length):
    image_string = ""
    # splitted_image_counts is a list of tuples (n_h, n_w)
    for idx, (n_h, n_w) in enumerate(splitted_image_counts):
        if len(splitted_image_counts) > 1:
            image_string += f"<image: {idx}>"
        for i in range(n_h):
            for j in range(n_w):
                # Use grid tokens if available, otherwise just use image tokens
                # This handles cases where we have more than 4x4 grid due to padding
                grid_token_name = f"r{i + 1}c{j + 1}"
                if hasattr(tokenizer, grid_token_name):
                    image_string += getattr(tokenizer, grid_token_name)
                # Always add the image tokens regardless
                image_string += tokenizer.image_token * mp_image_token_length
    return image_string
