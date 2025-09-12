from dataclasses import dataclass, field


def round_up_to_multiple(x: int, N: int = 128) -> int:
    """
    Rounds up integer x to the nearest multiple of N (default: 128).
    This improves GPU efficiency for matrix operations.
    """
    return ((x + N - 1) // N) * N


@dataclass
class VLMConfig:
    # === Vision Encoder Configuration ===
    # Model selection
    vit_architecture: str = "siglip"  # Options: 'siglip', 'dinov3'
    vit_model_type: str = "google/siglip2-base-patch16-512"

    # Common vision parameters
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 4 * vit_hidden_dim
    vit_patch_size: int = 16
    vit_img_size: int = 512
    vit_channels: int = 3  # Number of input channels (RGB)
    vit_n_heads: int = 12
    vit_dropout: float = 0.0
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False

    # DINOv3-specific parameters
    vit_num_registers: int = 0  # Set to 4 for DINOv3
    vit_use_swiglu: bool = False  # True for DINOv3
    vit_use_rope: bool = False  # DINOv3 doesn't actually use RoPE
    vit_use_sincos_pos: bool = False  # DINOv3 uses sin/cos embeddings
    vit_rope_base: float = 10000.0  # RoPE base for vision (if used)
    vit_layer_scale: bool = False  # True for DINOv3
    vit_layer_scale_init: float = 1.0  # LayerScale initialization
    vit_drop_path_rate: float = 0.0  # Stochastic depth rate
    vit_rope_augment: bool = (
        True  # Enable RoPE augmentations during training (DINOv3 paper)
    )

    # === Language Model Configuration ===
    # Model selection
    lm_architecture: str = "llama"  # Options: 'llama', 'gemma'
    lm_model_type: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    lm_tokenizer: str = "HuggingFaceTB/SmolLM2-360M-Instruct"

    # Common LM parameters
    lm_hidden_dim: int = 960
    lm_inter_dim: int = 2560
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_base_vocab_size: int = 49152
    extra_token_amount: int = (
        17  # Number of extra tokens for the VLM (image start, image end, image token)
    )
    lm_vocab_size: int = field(
        init=False
    )  # Will be computed with rounding for GPU efficiency
    lm_n_heads: int = 15
    lm_n_kv_heads: int = 5
    lm_head_dim: int = None  # Optional custom head dimension (e.g., Gemma-3 uses 256)
    lm_dropout: float = 0.0
    lm_n_blocks: int = 32
    lm_attn_scaling: float = 1.0
    lm_max_length: int = 1024
    lm_use_tokens: bool = False  # Decide if the LM expects tokens or embeddings as input (if using as a backbone for the VLM, set to False)
    lm_tie_weights: bool = True  # Decide if you want to tie the LM Head weight to the token embedding weights
    lm_chat_template: str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    # === Modality Projector Configuration ===
    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = (
        64  # 1 for DINOv3 (dynamic grids), >1 for SigLIP (fixed grids)
    )
    mp_handle_special_tokens: bool = False  # True for DINOv3 to remove registers

    max_img_size: int = 1024

    # Token budget control
    vlm_max_image_tokens: int = (
        1024  # Maximum tokens per image to prevent context explosion
    )

    vlm_extra_tokens: dict[str, str] = field(
        default_factory=lambda: {
            "image_token": "<|image|>",
            "r1c1": "<row_1_col_1>",
            "r1c2": "<row_1_col_2>",
            "r1c3": "<row_1_col_3>",
            "r1c4": "<row_1_col_4>",
            "r2c1": "<row_2_col_1>",
            "r2c2": "<row_2_col_2>",
            "r2c3": "<row_2_col_3>",
            "r2c4": "<row_2_col_4>",
            "r3c1": "<row_3_col_1>",
            "r3c2": "<row_3_col_2>",
            "r3c3": "<row_3_col_3>",
            "r3c4": "<row_3_col_4>",
            "r4c1": "<row_4_col_1>",
            "r4c2": "<row_4_col_2>",
            "r4c3": "<row_4_col_3>",
            "r4c4": "<row_4_col_4>",
        }
    )
    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str = "checkpoints"
    hf_repo_name: str = None  # Set to repo name to push to HuggingFace Hub

    def __post_init__(self):
        # Round vocab size to nearest multiple of 128 for GPU efficiency
        self.lm_vocab_size = round_up_to_multiple(
            self.lm_base_vocab_size + self.extra_token_amount
        )


@dataclass
class TrainConfig:
    lr_mp: float = 0.00512
    lr_backbones: float = 5e-5
    data_cutoff_idx: int = None
    val_ratio: float = 0.025
    batch_size: int = 2
    gradient_accumulation_steps: int = 32
    max_grad_norm: float = 1.0
    eval_in_epochs: bool = True
    eval_interval: int = 500
    validation_steps: int = (
        None  # Run validation every N steps (overrides eval_interval if set)
    )
    max_validation_samples: int = None  # Limit validation to N samples (None = use all)
    stats_log_interval: int = gradient_accumulation_steps * 25
    save_checkpoint_steps: int = (
        None  # Save checkpoint every N steps (in addition to best model)
    )
    console_log_interval: int = 25  # Print train metrics to console every N steps
    max_training_steps: int = 5000
    max_images_per_example: int = 4
    max_images_per_knapsack: int = 18
    max_sample_length: int = 1024
    compile: bool = False
    resume_from_vlm_checkpoint: bool = False  # Indicate if the training should be resumed from a checkpoint of the whole VLM or you want to start from scratch
    freeze_vision_encoder: bool = False  # Freeze vision encoder weights during training (Locked-image Text tuning)
    train_dataset_path: str = "HuggingFaceM4/the_cauldron"
    train_dataset_name: tuple[str, ...] = ("all",)
    wandb_entity: str = None  # Leave as None to use your default wandb entity
    log_wandb: bool = True
    wandb_log_steps: int = 5  # Log to wandb every N steps (default 5 for efficiency)
    use_lmms_eval: bool = (
        False  # Use lmms-eval for evaluation (expensive, disabled by default)
    )
    lmms_eval_tasks: str = "mmstar,mmmu,ocrbench,textvqa"  # Pass additional task as one string, seperated by commas without spaces (e.g. 'mmstar,mmmu,ocrbench')
    lmms_eval_limit: int = 2000
    lmms_eval_batch_size: int = 128
    num_workers: int = (
        2  # Number of DataLoader worker threads (set to 0 to disable multiprocessing)
    )
    max_threads: int = (
        4  # Maximum number of threads for torch operations (None = no limit)
    )


# Preset configurations
def get_original_small_config():
    """Returns config for original smaller models: SigLIP-B/16-224-85M + SmolLM2-135M (222M total)"""
    return VLMConfig(
        # SigLIP-B/16-224-85M settings
        vit_architecture="siglip",
        vit_model_type="google/siglip-base-patch16-224",  # Original smaller SigLIP
        vit_img_size=224,  # 224x224 images
        vit_patch_size=16,
        vit_hidden_dim=768,  # Base model dimensions
        vit_inter_dim=3072,  # 4 * 768
        vit_n_heads=12,
        vit_n_blocks=12,
        vit_cls_flag=False,  # SigLIP doesn't use CLS token
        # SmolLM2-135M settings
        lm_architecture="llama",
        lm_model_type="HuggingFaceTB/SmolLM2-135M",  # Original smaller SmolLM
        lm_tokenizer="HuggingFaceTB/SmolLM2-135M",
        lm_hidden_dim=576,  # SmolLM2-135M dimensions
        lm_inter_dim=1536,  # Intermediate dimension
        lm_n_heads=9,
        lm_n_kv_heads=3,
        lm_n_blocks=30,
        lm_base_vocab_size=49152,
        lm_max_position_embeddings=8192,
        lm_rms_eps=1e-5,
        lm_re_base=100000,
        # Modality projector - for 224x224 images
        mp_pixel_shuffle_factor=2,  # Since 224/16 = 14, need factor of 2
        mp_image_token_length=49,  # (14/2)^2 = 49
        mp_handle_special_tokens=False,
        # Override max_img_size to match input
        max_img_size=224,  # This preset uses smaller images for compatibility
    )


def get_dinov3_gemma_config():
    """Returns config for DINOv3 + Gemma - dimensions auto-detected from models"""
    return VLMConfig(
        # DINOv3 settings - from actual model config
        vit_architecture="dinov3",
        vit_model_type="facebook/dinov3-vits16plus-pretrain-lvd1689m",  # Real DINOv3
        vit_cls_flag=True,
        vit_num_registers=4,  # DINOv3 has 4 register tokens
        vit_use_swiglu=True,  # DINOv3 DOES use gated MLP (SwiGLU)
        vit_use_rope=True,  # DINOv3 paper recommends RoPE for improved spatial encoding
        vit_use_sincos_pos=True,  # DINOv3 uses sin/cos position embeddings
        vit_layer_scale=True,  # DINOv3 uses LayerScale
        vit_layer_scale_init=1.0,
        vit_drop_path_rate=0.0,  # No dropout in pretrained model
        vit_rope_augment=True,  # Enable RoPE augmentations during training
        vit_img_size=1024,  # DINOv3 can handle high-res images with RoPE
        vit_patch_size=16,
        # Dimensions will be auto-detected from model
        # Gemma settings
        lm_architecture="gemma",
        lm_model_type="google/gemma-3-270m-it",  # Real Gemma 3 270M
        lm_tokenizer="google/gemma-3-270m-it",
        # Dimensions will be auto-detected from model config
        # Just set vocab to handle extra tokens
        lm_base_vocab_size=262144,  # Gemma-3 actual vocab
        extra_token_amount=17,
        # lm_vocab_size will be auto-computed and rounded in __post_init__
        # Modality projector - DINOv3 uses dynamic grids
        mp_handle_special_tokens=True,
        mp_pixel_shuffle_factor=2,
        mp_image_token_length=1,  # DINOv3: 1 token per grid cell (dynamic)
    )
