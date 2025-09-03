# Modality Projection from Vision to Language
import torch.nn as nn


class ModalityProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Handle special tokens based on vision encoder
        self.handle_special_tokens = (
            cfg.mp_handle_special_tokens
            if hasattr(cfg, "mp_handle_special_tokens")
            else False
        )
        self.num_registers = (
            cfg.vit_num_registers
            if self.handle_special_tokens and hasattr(cfg, "vit_num_registers")
            else 0
        )
        self.has_cls = cfg.vit_cls_flag if self.handle_special_tokens else False

        # Calculate input dimension
        self.input_dim = cfg.vit_hidden_dim * (cfg.mp_pixel_shuffle_factor**2)
        self.output_dim = cfg.lm_hidden_dim
        self.scale_factor = cfg.mp_pixel_shuffle_factor

        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L1281
    def pixel_shuffle(self, x):
        bsz, seq, embed_dim = x.size()
        seq_root = int(seq**0.5)
        assert (
            seq_root**2 == seq
        )  # Sequence length must be a perfect square for pixel shuffle
        assert (
            seq_root % self.scale_factor == 0
        )  # Sequence root must be divisible by scale factor

        height = width = seq_root
        x = x.view(bsz, height, width, embed_dim)
        h_out = height // self.scale_factor
        w_out = width // self.scale_factor

        x = x.reshape(
            bsz, h_out, self.scale_factor, w_out, self.scale_factor, embed_dim
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(bsz, h_out * w_out, embed_dim * self.scale_factor**2)

        return x

    def forward(self, x):
        bsz, seq_len, embed_dim = x.size()

        # Remove special tokens if needed (for DINOv3)
        if self.handle_special_tokens:
            # Calculate starting index for patch tokens
            start_idx = 0
            if self.has_cls:
                start_idx += 1  # Skip CLS token
            if self.num_registers > 0:
                start_idx += self.num_registers  # Skip register tokens

            # Keep only patch tokens
            x = x[:, start_idx:, :]

        # Apply pixel shuffle
        x = self.pixel_shuffle(x)

        # Project to LM dimension
        x = self.proj(x)

        return x
