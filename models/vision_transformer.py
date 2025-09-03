import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionRotaryEmbedding(nn.Module):
    """Rotary embeddings for vision transformer (DINOv3)"""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return torch.cos(emb), torch.sin(emb)


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L245
class ViTPatchEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.img_size = cfg.vit_img_size
        self.patch_size = cfg.vit_patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.cls_flag = cfg.vit_cls_flag
        self.embd_dim = cfg.vit_hidden_dim
        self.use_rope = cfg.vit_use_rope if hasattr(cfg, "vit_use_rope") else False
        self.num_registers = (
            cfg.vit_num_registers if hasattr(cfg, "vit_num_registers") else 0
        )

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

        # Position embeddings (only for non-RoPE models)
        if not self.use_rope:
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
        x = self.conv(x)  # extract patches
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

        # Add position embeddings for non-RoPE models
        if not self.use_rope:
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
        self.use_rope = cfg.vit_use_rope if hasattr(cfg, "vit_use_rope") else False

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
            self.rope = VisionRotaryEmbedding(
                self.head_dim,
                base=cfg.vit_rope_base if hasattr(cfg, "vit_rope_base") else 10000.0,
            )

        # Use scaled dot product attention if available
        self.sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.sdpa:
            print(
                "Warning: scaled dot product attention not available. Using standard attention in ViT."
            )

    def apply_rope(self, q, k, seq_len):
        """Apply rotary position embeddings"""
        cos, sin = self.rope(seq_len)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply rotation
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        return q_rot, k_rot

    def rotate_half(self, x):
        """Helper for RoPE"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
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

        # Apply RoPE if enabled
        if self.use_rope:
            q, k = self.apply_rope(q, k, T)

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
    """SwiGLU FFN for DINOv3"""

    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.vit_hidden_dim, cfg.vit_inter_dim, bias=True)
        self.w2 = nn.Linear(cfg.vit_inter_dim, cfg.vit_hidden_dim, bias=True)
        self.w3 = nn.Linear(cfg.vit_hidden_dim, cfg.vit_inter_dim, bias=True)
        self.dropout = nn.Dropout(cfg.vit_dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


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
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.attn = ViTMultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)

        # Choose FFN type based on config
        if hasattr(cfg, "vit_use_swiglu") and cfg.vit_use_swiglu:
            self.mlp = ViTSwiGLUFFN(cfg)
        else:
            self.mlp = ViTMLP(cfg)  # Original MLP

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_embedding = ViTPatchEmbeddings(cfg)
        self.cls_flag = cfg.vit_cls_flag
        self.dropout = nn.Dropout(cfg.vit_dropout)
        self.blocks = nn.ModuleList([ViTBlock(cfg) for _ in range(cfg.vit_n_blocks)])
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
        for block in self.blocks:
            x = block(x)

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
        is_dinov3 = "dinov" in cfg.vit_model_type.lower()

        # Load HF config
        if is_dinov3:
            hf_config = AutoConfig.from_pretrained(cfg.vit_model_type)
            # Update config for DINOv3/DINOv2
            cfg.vit_architecture = "dinov3"
            cfg.vit_hidden_dim = hf_config.hidden_size
            # DINOv2 config doesn't have intermediate_size, calculate it
            cfg.vit_inter_dim = getattr(hf_config, "intermediate_size", hf_config.hidden_size * 4)
            cfg.vit_n_heads = hf_config.num_attention_heads
            cfg.vit_n_blocks = hf_config.num_hidden_layers
            cfg.vit_patch_size = hf_config.patch_size
            cfg.vit_img_size = hf_config.image_size
            cfg.vit_cls_flag = True
            cfg.vit_num_registers = getattr(hf_config, "num_register_tokens", 0)
            cfg.vit_use_swiglu = getattr(hf_config, "use_swiglu_ffn", False)
            cfg.vit_use_rope = False  # DINOv2 doesn't use RoPE
            cfg.vit_ln_eps = getattr(hf_config, "layer_norm_eps", 1e-6)
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
            # DINOv3 weight mapping
            mapping = _get_dinov3_mapping(cfg)
        else:
            # SigLIP weight mapping
            mapping = {
                "vision_model.embeddings.patch_embedding.weight": "patch_embedding.conv.weight",
                "vision_model.embeddings.patch_embedding.bias": "patch_embedding.conv.bias",
                "vision_model.embeddings.position_embedding.weight": "patch_embedding.position_embedding",
                "vision_model.post_layernorm.weight": "layer_norm.weight",
                "vision_model.post_layernorm.bias": "layer_norm.bias",
            }

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
            mapping[f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight"] = (
                f"blocks.{i}.attn.out_proj.weight"
            )
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
            if not is_dinov3:  # Only for SigLIP - DINOv2 has QKV already concatenated
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
                    except:
                        # Some models might not have this, skip
                        pass

        model.load_state_dict(sd)
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
