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

        # Store original vit_img_size for reference
        self.original_vit_img_size = cfg.vit_img_size

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

    def pixel_shuffle_2d(self, x, gh, gw):
        """Pixel shuffle for rectangular grids.

        When processing padded batches, gh and gw should be the ACTUAL
        (unpadded) grid dimensions, and we extract only those tokens.
        """
        bsz, seq, embed_dim = x.size()

        # The expected sequence should match the grid dimensions passed in
        expected_seq = int(gh) * int(gw)

        # If we have more tokens than expected (due to padding), take only the real ones
        if seq > expected_seq:
            # Extract only the real (non-padded) tokens
            # Assuming padding is at the end (right and bottom)
            x = x[:, :expected_seq, :]
            seq = expected_seq

        assert seq == expected_seq, (
            f"seq {seq} != expected {expected_seq} (gh={gh}, gw={gw})"
        )
        s = self.scale_factor
        assert gh % s == 0 and gw % s == 0, (
            f"Grid dimensions (gh={gh}, gw={gw}) must be divisible by scale factor {s}"
        )

        # Calculate output dimensions
        ho, wo = gh // s, gw // s

        # Reshape and permute for pixel shuffle
        x = x.view(bsz, gh, gw, embed_dim)
        x = x.view(bsz, ho, s, wo, s, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(bsz, ho * wo, embed_dim * s * s)

        return x

    def forward(self, x, gh=None, gw=None):
        bsz, seq_len, embed_dim = x.size()

        # Remove special tokens if needed
        if self.handle_special_tokens:
            start_idx = 0
            if self.has_cls:
                start_idx += 1
            if self.num_registers > 0:
                start_idx += self.num_registers
            x = x[:, start_idx:, :]

        # Apply pixel shuffle
        if gh is not None and gw is not None:
            # Use 2D pixel shuffle for rectangular grids (DINOv3)
            x = self.pixel_shuffle_2d(x, gh, gw)
        else:
            # Use original square pixel shuffle (SigLIP)
            x = self.pixel_shuffle(x)

        x = self.proj(x)
        return x
