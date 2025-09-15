import torchvision.transforms as transforms
from torchvision.transforms import Normalize
from transformers import AutoTokenizer

from data.custom_transforms import DynamicResize, SplitImage
from models.grid_abstraction import DINOv3Grid

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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


def get_image_processor(
    max_img_size,
    splitted_image_size,
    single_image_mode=False,
    vit_patch_size=16,
    pixel_shuffle_factor=2,
    allow_upscale=True,
):
    """
    Create image processor.

    Args:
        max_img_size: Maximum size for image dimension
        splitted_image_size: Target size for each image/patch
        single_image_mode: If True, use aspect-preserving resize for DINOv3
        vit_patch_size: Patch size for vision transformer
        pixel_shuffle_factor: Factor for pixel shuffle downsampling
        allow_upscale: Whether to allow upscaling images
        grid_factory: Optional GridFactory for token calculation (used internally)
    """
    if single_image_mode:
        # DINOv3 path: aspect-preserving resize with proper grid calculation
        def process_single_image(pil_image):
            # Create transform pipeline
            # For DINOv3, we need dimensions divisible by patch_size * pixel_shuffle_factor
            effective_patch_size = vit_patch_size * pixel_shuffle_factor
            transform = transforms.Compose(
                [
                    DynamicResize(
                        patch_size=effective_patch_size,
                        max_side_len=max_img_size,
                        allow_upscale=allow_upscale,
                    ),
                    transforms.ToTensor(),
                    Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )

            # Apply transforms
            x = transform(pil_image)  # C, H, W
            _, H, W = x.shape

            # Calculate patch grid dimensions
            Hp, Wp = H // vit_patch_size, W // vit_patch_size

            # Calculate final grid after pixel shuffle
            s = pixel_shuffle_factor
            assert Hp % s == 0 and Wp % s == 0, (
                f"Patch grid (Hp={Hp}, Wp={Wp}) must be divisible by pixel_shuffle_factor={s}"
            )
            Gh, Gw = Hp // s, Wp // s

            # Return tensor and DINOv3Grid object
            grid = DINOv3Grid(Hp=Hp, Wp=Wp, Gh=Gh, Gw=Gw)
            return x, grid.to_dict()  # Keep dict format for backward compatibility

        return process_single_image
    else:
        # For SigLIP: split into multiple sub-images with normalization
        return transforms.Compose(
            [
                DynamicResize(
                    patch_size=splitted_image_size,
                    max_side_len=max_img_size,
                    allow_upscale=allow_upscale,
                ),
                transforms.ToTensor(),
                Normalize(IMAGENET_MEAN, IMAGENET_STD),
                SplitImage(patch_size=splitted_image_size),
            ]
        )


def get_image_string(
    tokenizer, splitted_image_counts, mp_image_token_length, config=None
):
    """
    Generate image token string based on actual grid configurations.

    Args:
        tokenizer: Tokenizer with image_token attribute
        splitted_image_counts: List of tuples (n_h, n_w) or grid info
        mp_image_token_length: Base token length per tile/grid cell
        config: VLMConfig for grid factory (optional)
    """
    from models.grid_abstraction import GridFactory

    image_string = ""
    for idx, grid_info in enumerate(splitted_image_counts):
        if len(splitted_image_counts) > 1:
            image_string += f"<image: {idx}>"

        # Create grid object to get actual token count
        if config is not None:
            grid = GridFactory.create_from_raw(grid_info, config)
            token_count = grid.get_token_count()
        else:
            # Fallback for backward compatibility
            n_h, n_w = grid_info if isinstance(grid_info, tuple) else (1, 1)
            if mp_image_token_length == 1:
                token_count = n_h * n_w
            else:
                token_count = n_h * n_w * mp_image_token_length

        # Emit placeholder tokens
        image_string += tokenizer.image_token * token_count
    return image_string
