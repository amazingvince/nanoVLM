import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic Depth / DropPath for DINOv3"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Work with diff dim tensors (2D/3D/4D)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class LayerScale(nn.Module):
    """LayerScale for DINOv3 - learnable per-channel scaling"""

    def __init__(self, dim: int, init_value: float = 1.0):
        super().__init__()
        self.lambda_param = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.lambda_param


def get_patches_center_coordinates(num_patches_h, num_patches_w, dtype, device):
    """
    Computes the 2D coordinates of the centers of image patches, normalized to the range [-1, +1].
    The center of each patch is exactly halfway between its top-left and bottom-right corners.
    """
    coords_h = torch.arange(0.5, num_patches_h, dtype=dtype, device=device)
    coords_w = torch.arange(0.5, num_patches_w, dtype=dtype, device=device)
    coords_h = coords_h / num_patches_h
    coords_w = coords_w / num_patches_w
    # (height, width, 2) -> (height * width, 2)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    coords = coords.flatten(0, 1)
    # Shift range [0, 1] to [-1, +1]
    coords = 2.0 * coords - 1.0
    return coords


def rotate_half(x):
    """Rotates half the hidden dims of the input for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, num_prefix_tokens=0):
    """Applies RoPE to query and key tensors, but only to patch tokens."""
    if num_prefix_tokens > 0:
        # Split prefix tokens (CLS + register) from patch tokens
        q_prefix, q_patches = q.split(
            (num_prefix_tokens, q.shape[-2] - num_prefix_tokens), dim=-2
        )
        k_prefix, k_patches = k.split(
            (num_prefix_tokens, k.shape[-2] - num_prefix_tokens), dim=-2
        )

        # Apply RoPE only to patch tokens
        q_patches = (q_patches * cos) + (rotate_half(q_patches) * sin)
        k_patches = (k_patches * cos) + (rotate_half(k_patches) * sin)

        # Concatenate back
        q = torch.cat((q_prefix, q_patches), dim=-2)
        k = torch.cat((k_prefix, k_patches), dim=-2)
    else:
        # Apply RoPE to all tokens
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

    return q, k


class DINOv3RoPEPositionEmbedding(nn.Module):
    """DINOv3-style RoPE position embeddings using patch center coordinates."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.base = getattr(cfg, "vit_rope_theta", 100.0)  # Default from DINOv3
        self.head_dim = cfg.vit_hidden_dim // cfg.vit_n_heads

        # Pre-compute inverse frequencies
        inv_freq = 1 / self.base ** torch.arange(
            0, 1, 4 / self.head_dim, dtype=torch.float32
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, num_patches_h, num_patches_w, dtype, device):
        """Generate cos/sin embeddings for the given patch grid."""
        # Get patch center coordinates normalized to [-1, +1]
        patch_coords = get_patches_center_coordinates(
            num_patches_h, num_patches_w, torch.float32, device
        )

        # Apply augmentations during training (as per DINOv3 paper)
        if (
            self.training
            and hasattr(self.cfg, "vit_rope_augment")
            and self.cfg.vit_rope_augment
        ):
            # Random shift (up to 10% of the range)
            shift = (torch.rand(2, device=device) - 0.5) * 0.2
            patch_coords = patch_coords + shift[None, :]

            # Random jitter (small noise)
            jitter = torch.randn_like(patch_coords) * 0.01
            patch_coords = patch_coords + jitter

            # Random rescale (95% to 105%)
            scale = 0.95 + torch.rand(1, device=device).item() * 0.1
            patch_coords = patch_coords * scale

        # Apply inverse frequencies to get angles
        # (height * width, 2, head_dim/4) -> (height * width, head_dim/2) -> (height * width, head_dim)
        angles = 2 * math.pi * patch_coords[:, :, None] * self.inv_freq[None, None, :]
        angles = angles.flatten(1, 2)
        angles = angles.tile(2)  # Repeat to match head_dim

        cos = torch.cos(angles).to(dtype=dtype)
        sin = torch.sin(angles).to(dtype=dtype)

        return cos, sin


class ViTPatchEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.img_size = cfg.vit_img_size
        self.patch_size = cfg.vit_patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.cls_flag = cfg.vit_cls_flag
        self.embd_dim = cfg.vit_hidden_dim
        self.use_dinov3_rope = getattr(cfg, "vit_architecture", "siglip") == "dinov3"
        self.num_registers = getattr(cfg, "vit_num_registers", 0)

        # Conv layer to extract the patches
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=self.embd_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        # Special tokens
        if self.cls_flag:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embd_dim))

        if self.num_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, self.num_registers, self.embd_dim)
            )

        # Position embeddings - DINOv3 uses RoPE in attention, SigLIP uses learned embeddings
        if not self.use_dinov3_rope:
            num_positions = self.num_patches
            if self.cls_flag:
                num_positions += 1
            if self.num_registers > 0:
                num_positions += self.num_registers
            self.position_embedding = nn.Parameter(
                torch.rand(1, num_positions, self.embd_dim)
            )

    def forward(self, x):
        B = x.shape[0]
        x = self.conv(x)  # extract patches -> B, C, Hp, Wp
        # Store the patch grid dimensions for RoPE
        Hp, Wp = x.shape[-2], x.shape[-1]
        self._last_hw = (Hp, Wp)
        x = x.flatten(2)  # flatten the patches into a single dimension
        x = x.transpose(1, 2)  # transpose to (batch_size, num_patches, hidden_dim)

        # Add CLS token if needed
        if self.cls_flag:
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        # Add register tokens if needed (after CLS, before patches)
        if self.num_registers > 0:
            register_tokens = self.register_tokens.expand(B, -1, -1)
            if self.cls_flag:
                # Insert registers after CLS token
                x = torch.cat((x[:, :1], register_tokens, x[:, 1:]), dim=1)
            else:
                # Insert registers at the beginning
                x = torch.cat((register_tokens, x), dim=1)

        # Add position embeddings (only for SigLIP, DINOv3 uses RoPE in attention)
        if not self.use_dinov3_rope:
            # Standard learned position embeddings (SigLIP)
            x = x + self.position_embedding

        return x


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L381
# https://github.com/karpathy/nanoGPT/blob/master/model.py#L29
class ViTMultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.vit_n_heads
        self.embd_dim = cfg.vit_hidden_dim
        self.head_dim = self.embd_dim // self.n_heads
        self.dropout = cfg.vit_dropout
        self.use_rope = getattr(cfg, "vit_use_rope", False)
        self.use_dinov3_rope = (
            self.use_rope and getattr(cfg, "vit_architecture", "siglip") == "dinov3"
        )

        assert self.embd_dim % self.n_heads == 0, (
            "embd_dim must be divisible by num_heads"
        )

        # Combined projections for all heads
        self.qkv_proj = nn.Linear(self.embd_dim, 3 * self.embd_dim, bias=True)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=True)

        # Dropout layers
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # RoPE if needed
        if self.use_rope:
            self.rope = DINOv3RoPEPositionEmbedding(
                self.head_dim,
                base=cfg.vit_rope_base if hasattr(cfg, "vit_rope_base") else 10000.0,
            )

        # Use scaled dot product attention if available
        self.sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.sdpa:
            print(
                "Warning: scaled dot product attention not available. Using standard attention in ViT."
            )

    def forward(self, x, position_embeddings=None):
        B, T, C = x.size()

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        # Reshape  [B, T, C] -> [B, T, n_heads, head_dim] and transpose -> [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # (B, n_heads, T, head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # (B, n_heads, T, head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # (B, n_heads, T, head_dim)

        # Apply DINOv3 RoPE if enabled
        if self.use_dinov3_rope and position_embeddings is not None:
            cos, sin = position_embeddings
            # Determine number of prefix tokens (CLS + registers)
            num_prefix = 1 if self.cfg.vit_cls_flag else 0
            num_prefix += getattr(self.cfg, "vit_num_registers", 0)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, num_prefix_tokens=num_prefix)

        if self.sdpa:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,  # ViT attention is bidirectional
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = (
                attn @ v
            )  # (B, n_heads, T, T) x (B, n_heads, T, head_dim) -> (B, n_heads, T, head_dim)

        # Transpose back from [B, n_heads, T, head_dim] to [B, T, n_heads * head_dim] and combine all heads to [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y


class ViTSwiGLUFFN(nn.Module):
    """SwiGLU FFN for DINOv3+ models"""

    def __init__(self, cfg):
        super().__init__()
        # Use HuggingFace naming convention for compatibility
        self.gate_proj = nn.Linear(cfg.vit_hidden_dim, cfg.vit_inter_dim, bias=True)
        self.down_proj = nn.Linear(cfg.vit_inter_dim, cfg.vit_hidden_dim, bias=True)
        self.up_proj = nn.Linear(cfg.vit_hidden_dim, cfg.vit_inter_dim, bias=True)
        self.dropout = nn.Dropout(cfg.vit_dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L453
class ViTMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.activation_fn = nn.GELU(approximate="tanh")
        self.fc1 = nn.Linear(cfg.vit_hidden_dim, cfg.vit_inter_dim)
        self.fc2 = nn.Linear(cfg.vit_inter_dim, cfg.vit_hidden_dim)
        self.dropout = nn.Dropout(cfg.vit_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# https://github.com/karpathy/nanoGPT/blob/master/model.py#L94
class ViTBlock(nn.Module):
    def __init__(self, cfg, drop_path_rate=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.attn = ViTMultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)

        # Choose FFN type based on config
        if hasattr(cfg, "vit_use_swiglu") and cfg.vit_use_swiglu:
            self.mlp = ViTSwiGLUFFN(cfg)
        else:
            self.mlp = ViTMLP(cfg)  # Original MLP

        # Add LayerScale if configured (DINOv3)
        self.use_layer_scale = getattr(cfg, "vit_layer_scale", False)
        if self.use_layer_scale:
            init_value = getattr(cfg, "vit_layer_scale_init", 1.0)
            self.layer_scale1 = LayerScale(cfg.vit_hidden_dim, init_value)
            self.layer_scale2 = LayerScale(cfg.vit_hidden_dim, init_value)

        # Add DropPath if configured (stochastic depth)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x, position_embeddings=None):
        # Attention block with optional LayerScale and DropPath
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale1(self.attn(self.ln1(x), position_embeddings))
            )
            x = x + self.drop_path(self.layer_scale2(self.mlp(self.ln2(x))))
        else:
            x = x + self.drop_path(self.attn(self.ln1(x), position_embeddings))
            x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class ViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_embedding = ViTPatchEmbeddings(cfg)
        self.cls_flag = cfg.vit_cls_flag
        self.dropout = nn.Dropout(cfg.vit_dropout)
        self.use_dinov3_rope = getattr(cfg, "vit_architecture", "siglip") == "dinov3"

        # Initialize DINOv3 RoPE if needed
        if self.use_dinov3_rope:
            self.rope_embeddings = DINOv3RoPEPositionEmbedding(cfg)

        # Create blocks with stochastic depth (linearly increasing drop rate)
        drop_path_rate = getattr(cfg, "vit_drop_path_rate", 0.0)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, cfg.vit_n_blocks)]
        self.blocks = nn.ModuleList(
            [ViTBlock(cfg, dpr[i]) for i in range(cfg.vit_n_blocks)]
        )
        self.layer_norm = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.dropout(x)

        # Generate DINOv3 RoPE embeddings if needed
        position_embeddings = None
        use_dinov3_rope = getattr(self.cfg, "vit_architecture", "siglip") == "dinov3"
        if use_dinov3_rope and hasattr(self, "rope_embeddings"):
            # Get the actual patch grid dimensions from patch embedding
            if hasattr(self.patch_embedding, "_last_hw"):
                num_patches_h, num_patches_w = self.patch_embedding._last_hw
            else:
                # Fallback to square grid if dimensions not available
                num_patches_h = num_patches_w = int(
                    math.sqrt(
                        (x.shape[1] - 1 - getattr(self.cfg, "vit_num_registers", 0))
                        if self.cls_flag
                        else x.shape[1] - getattr(self.cfg, "vit_num_registers", 0)
                    )
                )
            cos, sin = self.rope_embeddings(
                num_patches_h, num_patches_w, x.dtype, x.device
            )
            # Expand for batch and heads
            cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, num_patches, head_dim)
            sin = sin.unsqueeze(0).unsqueeze(0)
            position_embeddings = (cos, sin)

        for block in self.blocks:
            x = block(x, position_embeddings)

        x = self.layer_norm(x)

        # Return full sequence for modality projector to handle
        return x

    # Load the model from a pretrained HuggingFace model (we don't want to have to train the Vision Backbone from scratch)
    @classmethod
    def from_pretrained(cls, cfg):
        import safetensors
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig

        # Detect model type
        is_dinov2 = "dinov2" in cfg.vit_model_type.lower()
        is_dinov3 = "dinov3" in cfg.vit_model_type.lower()

        # Load HF config
        if is_dinov2 or is_dinov3:
            hf_config = AutoConfig.from_pretrained(cfg.vit_model_type)
            # Update config for DINOv3/DINOv2
            cfg.vit_architecture = "dinov3"
            cfg.vit_hidden_dim = hf_config.hidden_size
            # DINOv2 config doesn't have intermediate_size, calculate it
            cfg.vit_inter_dim = getattr(
                hf_config, "intermediate_size", hf_config.hidden_size * 4
            )
            cfg.vit_n_heads = hf_config.num_attention_heads
            cfg.vit_n_blocks = hf_config.num_hidden_layers
            cfg.vit_patch_size = hf_config.patch_size
            cfg.vit_img_size = hf_config.image_size
            cfg.vit_cls_flag = True
            cfg.vit_num_registers = getattr(hf_config, "num_register_tokens", 0)
            # DINOv3 uses "use_gated_mlp" in config, not "use_swiglu_ffn"
            cfg.vit_use_swiglu = getattr(
                hf_config, "use_gated_mlp", getattr(hf_config, "use_swiglu_ffn", False)
            )
            # DINOv3 uses RoPE with patch-center coordinates for positional encoding
            cfg.vit_use_rope = True
            cfg.vit_use_sincos_pos = False  # DINOv3 uses RoPE, not sin/cos embeddings
            cfg.vit_ln_eps = getattr(hf_config, "layer_norm_eps", 1e-6)

            # DINOv3 features
            cfg.vit_layer_scale = True  # DINOv3 uses LayerScale
            cfg.vit_layer_scale_init = getattr(hf_config, "layerscale_value", 1.0)
            cfg.vit_drop_path_rate = getattr(hf_config, "drop_path_rate", 0.0)
        else:
            # SigLIP config
            from transformers import SiglipVisionConfig

            hf_config = SiglipVisionConfig.from_pretrained(cfg.vit_model_type)
            cfg.vit_architecture = "siglip"
            cfg.vit_dropout = hf_config.attention_dropout
            cfg.vit_hidden_dim = hf_config.hidden_size
            cfg.vit_img_size = hf_config.image_size
            cfg.vit_inter_dim = hf_config.intermediate_size
            cfg.vit_ln_eps = hf_config.layer_norm_eps
            cfg.vit_n_heads = hf_config.num_attention_heads
            cfg.vit_n_blocks = hf_config.num_hidden_layers
            cfg.vit_patch_size = hf_config.patch_size

        model = cls(cfg)
        safetensors_file = hf_hub_download(
            repo_id=cfg.vit_model_type, filename="model.safetensors"
        )

        sd = model.state_dict()

        if is_dinov3:
            # DINOv3 weight mapping - HF keys -> our keys
            mapping = {
                "embeddings.patch_embeddings.weight": "patch_embedding.conv.weight",
                "embeddings.patch_embeddings.bias": "patch_embedding.conv.bias",
                # Note: cls_token and register_tokens don't exist in our model
                # "embeddings.cls_token": "patch_embedding.cls_token",
                # "embeddings.register_tokens": "patch_embedding.register_tokens",
                "norm.weight": "layer_norm.weight",
                "norm.bias": "layer_norm.bias",
            }
            # Add layer mappings for DINOv3
            for i in range(cfg.vit_n_blocks):
                # Layer norms
                mapping[f"layer.{i}.norm1.weight"] = f"blocks.{i}.ln1.weight"
                mapping[f"layer.{i}.norm1.bias"] = f"blocks.{i}.ln1.bias"
                mapping[f"layer.{i}.norm2.weight"] = f"blocks.{i}.ln2.weight"
                mapping[f"layer.{i}.norm2.bias"] = f"blocks.{i}.ln2.bias"

                # LayerScale parameters
                if cfg.vit_layer_scale:
                    mapping[f"layer.{i}.layer_scale1.lambda1"] = (
                        f"blocks.{i}.layer_scale1.lambda_param"
                    )
                    mapping[f"layer.{i}.layer_scale2.lambda1"] = (
                        f"blocks.{i}.layer_scale2.lambda_param"
                    )

                # Attention - q,k,v handled separately in QKV concatenation
                mapping[f"layer.{i}.attention.o_proj.weight"] = (
                    f"blocks.{i}.attn.out_proj.weight"
                )
                mapping[f"layer.{i}.attention.o_proj.bias"] = (
                    f"blocks.{i}.attn.out_proj.bias"
                )

                # MLP - Check if this DINOv3 model uses gated MLP (SwiGLU)
                if cfg.vit_use_swiglu:
                    # DINOv3+ models use SwiGLU with gate_proj
                    # mapping[hf_key] = our_key
                    mapping[f"layer.{i}.mlp.gate_proj.weight"] = (
                        f"blocks.{i}.mlp.gate_proj.weight"
                    )
                    mapping[f"layer.{i}.mlp.gate_proj.bias"] = (
                        f"blocks.{i}.mlp.gate_proj.bias"
                    )
                    mapping[f"layer.{i}.mlp.up_proj.weight"] = (
                        f"blocks.{i}.mlp.up_proj.weight"
                    )
                    mapping[f"layer.{i}.mlp.up_proj.bias"] = (
                        f"blocks.{i}.mlp.up_proj.bias"
                    )
                    mapping[f"layer.{i}.mlp.down_proj.weight"] = (
                        f"blocks.{i}.mlp.down_proj.weight"
                    )
                    mapping[f"layer.{i}.mlp.down_proj.bias"] = (
                        f"blocks.{i}.mlp.down_proj.bias"
                    )
                else:
                    # Regular DINOv3 models use standard MLP
                    # mapping[hf_key] = our_key
                    mapping[f"layer.{i}.mlp.up_proj.weight"] = (
                        f"blocks.{i}.mlp.fc1.weight"
                    )
                    mapping[f"layer.{i}.mlp.up_proj.bias"] = f"blocks.{i}.mlp.fc1.bias"
                    mapping[f"layer.{i}.mlp.down_proj.weight"] = (
                        f"blocks.{i}.mlp.fc2.weight"
                    )
                    mapping[f"layer.{i}.mlp.down_proj.bias"] = (
                        f"blocks.{i}.mlp.fc2.bias"
                    )
        elif is_dinov2:
            # DINOv2 weight mapping
            mapping = {
                "embeddings.patch_embeddings.projection.weight": "patch_embedding.conv.weight",
                "embeddings.patch_embeddings.projection.bias": "patch_embedding.conv.bias",
            }
        else:
            # SigLIP weight mapping
            mapping = {
                "vision_model.embeddings.patch_embedding.weight": "patch_embedding.conv.weight",
                "vision_model.embeddings.patch_embedding.bias": "patch_embedding.conv.bias",
                "vision_model.embeddings.position_embedding.weight": "patch_embedding.position_embedding",
                "vision_model.post_layernorm.weight": "layer_norm.weight",
                "vision_model.post_layernorm.bias": "layer_norm.bias",
            }

        if not is_dinov3 and not is_dinov2:
            # Only add SigLIP layer mappings
            for i in range(cfg.vit_n_blocks):
                # Layer norms
                mapping[f"vision_model.encoder.layers.{i}.layer_norm1.weight"] = (
                    f"blocks.{i}.ln1.weight"
                )
                mapping[f"vision_model.encoder.layers.{i}.layer_norm1.bias"] = (
                    f"blocks.{i}.ln1.bias"
                )
                mapping[f"vision_model.encoder.layers.{i}.layer_norm2.weight"] = (
                    f"blocks.{i}.ln2.weight"
                )
                mapping[f"vision_model.encoder.layers.{i}.layer_norm2.bias"] = (
                    f"blocks.{i}.ln2.bias"
                )

                # MLP
                mapping[f"vision_model.encoder.layers.{i}.mlp.fc1.weight"] = (
                    f"blocks.{i}.mlp.fc1.weight"
                )
                mapping[f"vision_model.encoder.layers.{i}.mlp.fc1.bias"] = (
                    f"blocks.{i}.mlp.fc1.bias"
                )
                mapping[f"vision_model.encoder.layers.{i}.mlp.fc2.weight"] = (
                    f"blocks.{i}.mlp.fc2.weight"
                )
                mapping[f"vision_model.encoder.layers.{i}.mlp.fc2.bias"] = (
                    f"blocks.{i}.mlp.fc2.bias"
                )

                # Output projection
                mapping[
                    f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight"
                ] = f"blocks.{i}.attn.out_proj.weight"
                mapping[f"vision_model.encoder.layers.{i}.self_attn.out_proj.bias"] = (
                    f"blocks.{i}.attn.out_proj.bias"
                )

        with safetensors.safe_open(
            filename=safetensors_file, framework="pt", device="cpu"
        ) as f:
            for hf_key, our_key in mapping.items():
                if hf_key in f.keys() and our_key in sd:
                    tensor = f.get_tensor(hf_key)
                    if tensor.shape == sd[our_key].shape:
                        sd[our_key].copy_(tensor)
                    else:
                        if "position_embedding" in hf_key:
                            sd[our_key].copy_(tensor.unsqueeze(0))
                        else:
                            print(
                                f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}"
                            )
                else:
                    if hf_key not in f.keys():
                        print(f"Warning: Key {hf_key} not found in safetensors file")
                    if our_key not in sd:
                        print(f"Warning: Key {our_key} not found in model state dict")

            # Manually handle QKV concatenation since our implementation combines Q, K, V into one
            if is_dinov3:
                # DINOv3 has separate Q, K, V - need to concatenate
                for i in range(model.cfg.vit_n_blocks):
                    try:
                        q_weight = f.get_tensor(f"layer.{i}.attention.q_proj.weight")
                        k_weight = f.get_tensor(f"layer.{i}.attention.k_proj.weight")
                        v_weight = f.get_tensor(f"layer.{i}.attention.v_proj.weight")

                        # Concatenate Q, K, V weights
                        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                        sd[f"blocks.{i}.attn.qkv_proj.weight"].copy_(qkv_weight)

                        # Same for biases if they exist
                        if f"layer.{i}.attention.q_proj.bias" in f.keys():
                            q_bias = f.get_tensor(f"layer.{i}.attention.q_proj.bias")
                            k_bias = torch.zeros_like(
                                q_bias
                            )  # K doesn't have bias in DINOv3
                            v_bias = f.get_tensor(f"layer.{i}.attention.v_proj.bias")
                            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                            sd[f"blocks.{i}.attn.qkv_proj.bias"].copy_(qkv_bias)
                    except Exception:
                        # Some models might not have this, skip
                        pass
            elif not is_dinov2:  # Only for SigLIP
                for i in range(model.cfg.vit_n_blocks):
                    q_weight = f.get_tensor(
                        f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight"
                    )
                    k_weight = f.get_tensor(
                        f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight"
                    )
                    v_weight = f.get_tensor(
                        f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight"
                    )

                    qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
                    sd[f"blocks.{i}.attn.qkv_proj.weight"].copy_(qkv_weight)

                    q_bias = f.get_tensor(
                        f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias"
                    )
                    k_bias = f.get_tensor(
                        f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias"
                    )
                    v_bias = f.get_tensor(
                        f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias"
                    )

                    qkv_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
                    sd[f"blocks.{i}.attn.qkv_proj.bias"].copy_(qkv_bias)
            else:  # For DINOv2 - handle QKV that's already concatenated
                for i in range(model.cfg.vit_n_blocks):
                    try:
                        qkv_weight = f.get_tensor(f"encoder.layer.{i}.attn.qkv.weight")
                        qkv_bias = f.get_tensor(f"encoder.layer.{i}.attn.qkv.bias")
                        sd[f"blocks.{i}.attn.qkv_proj.weight"].copy_(qkv_weight)
                        sd[f"blocks.{i}.attn.qkv_proj.bias"].copy_(qkv_bias)
                    except Exception:
                        # Some models might not have this, skip
                        pass

        model.load_state_dict(sd)

        # Initialize DINOv3 RoPE embeddings after loading if needed
        if cfg.vit_architecture == "dinov3" and not hasattr(model, "rope_embeddings"):
            model.rope_embeddings = DINOv3RoPEPositionEmbedding(cfg)

        print(
            f"Successfully loaded {cfg.vit_model_type} weights from safetensors. Model has {sum(p.numel() for p in model.parameters()):,} parameters."
        )
        return model


def _get_dinov3_mapping(cfg):
    """Get DINOv2/v3 weight mapping"""
    mapping = {
        "embeddings.patch_embeddings.projection.weight": "patch_embedding.conv.weight",
        "embeddings.patch_embeddings.projection.bias": "patch_embedding.conv.bias",
        "embeddings.cls_token": "patch_embedding.cls_token",
        "embeddings.position_embeddings": "patch_embedding.position_embedding",
        "layernorm.weight": "layer_norm.weight",
        "layernorm.bias": "layer_norm.bias",
    }

    # Add register tokens if present (DINOv3)
    if cfg.vit_num_registers > 0:
        mapping["embeddings.register_tokens"] = "patch_embedding.register_tokens"

    # Add layer mappings - DINOv2 uses different naming
    for i in range(cfg.vit_n_blocks):
        prefix = f"encoder.layer.{i}."
        our_prefix = f"blocks.{i}."

        mapping.update(
            {
                # Layer norms
                f"{prefix}norm1.weight": f"{our_prefix}ln1.weight",
                f"{prefix}norm1.bias": f"{our_prefix}ln1.bias",
                f"{prefix}norm2.weight": f"{our_prefix}ln2.weight",
                f"{prefix}norm2.bias": f"{our_prefix}ln2.bias",
                # Attention
                f"{prefix}attn.proj.weight": f"{our_prefix}attn.out_proj.weight",
                f"{prefix}attn.proj.bias": f"{our_prefix}attn.out_proj.bias",
                # MLP layers - DINOv2 uses fc1/fc2
                f"{prefix}mlp.fc1.weight": f"{our_prefix}mlp.fc1.weight",
                f"{prefix}mlp.fc1.bias": f"{our_prefix}mlp.fc1.bias",
                f"{prefix}mlp.fc2.weight": f"{our_prefix}mlp.fc2.weight",
                f"{prefix}mlp.fc2.bias": f"{our_prefix}mlp.fc2.bias",
            }
        )

    return mapping
