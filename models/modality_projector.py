# Modality Projection from Vision to Language
import torch
import torch.nn as nn


class ModalityProjector(nn.Module):
    """Projects vision features to language embedding space using pixel shuffle and linear projection.

    :param cfg: VLMConfig containing modality projection parameters
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Get vision encoder hidden dim (may be updated by encoder factory)
        vision_hidden_dim = cfg.vit_hidden_dim

        self.input_dim = vision_hidden_dim * (cfg.mp_pixel_shuffle_factor**2)
        self.output_dim = cfg.lm_hidden_dim
        self.scale_factor = cfg.mp_pixel_shuffle_factor

        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize linear layer weights with normal distribution.
        
        :param module: Module to initialize
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L1281
    def pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        """Rearrange spatial patches to reduce sequence length while increasing feature dimension.
        
        :param x: Input tensor [batch_size, seq_len, embed_dim] where seq_len must be perfect square
        :return: Shuffled tensor [batch_size, seq_len/(scale_factor^2), embed_dim*(scale_factor^2)]
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project vision features to language embedding space.
        
        :param x: Vision features [batch_size, num_patches, vit_hidden_dim]
        :return: Language embeddings [batch_size, reduced_patches, lm_hidden_dim]
        """
        x = self.pixel_shuffle(x)
        x = self.proj(x)

        return x
