# Vision Encoders Documentation

## Overview

nanoVLM now supports multiple vision encoders through a flexible abstraction layer. This allows you to easily switch between different vision backbones while maintaining the same training and inference pipeline.

## Supported Encoders

### SigLIP (Default)
- **Model**: `google/siglip2-base-patch16-512`
- **Parameters**: 86M
- **Hidden Dimension**: 768
- **Patch Size**: 16×16
- **Default Input Size**: 512×512
- **Special Features**: No CLS token, uses all patch embeddings

### DINOv3-small
- **Model**: `facebook/dinov3-vits16plus-pretrain-lvd1689m`
- **Parameters**: 30M
- **Hidden Dimension**: 384
- **Patch Size**: 16×16
- **Default Input Size**: 518×518
- **Special Features**:
  - Uses SwiGLU FFN (gated MLP)
  - 2D RoPE position embeddings
  - Register tokens (4 special tokens)
  - CLS token for global representation
  - LayerScale and DropPath

### DINOv3-base
- **Model**: `facebook/dinov3-vitb16-pretrain-lvd1689m`
- **Parameters**: 86M
- **Hidden Dimension**: 768
- **Patch Size**: 16×16
- **Default Input Size**: 518×518
- **Special Features**:
  - Standard GELU MLP (not gated)
  - 2D RoPE position embeddings
  - Register tokens (4 special tokens)
  - CLS token for global representation
  - LayerScale and DropPath

## Usage

### Training with Different Encoders

```bash
# Default SigLIP encoder
python train.py

# DINOv3-small with frozen encoder (recommended)
python train.py --vision_encoder_type dinov3-small --freeze_vision_encoder

# DINOv3-base with frozen encoder (recommended)
python train.py --vision_encoder_type dinov3-base --freeze_vision_encoder
```

### Why Freeze DINOv3?

DINOv3 models are specifically designed and pretrained with self-supervised objectives that create strong visual representations. Meta recommends freezing these encoders during VLM training because:

1. **Preserves learned representations**: DINOv3's pretraining creates robust features that shouldn't be modified
2. **Training stability**: Frozen encoders lead to more stable gradient flow (10-20 grad norm vs 15-300)
3. **Faster training**: Higher throughput due to fewer gradient computations
4. **Better generalization**: Prevents overfitting the vision encoder to the VLM task

### Programmatic Usage

```python
from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel

# Configure with DINOv3-small
cfg = VLMConfig()
cfg.vision_encoder_type = "dinov3-small"

# Create model
model = VisionLanguageModel(cfg, load_backbone=True)

# Optionally freeze the vision encoder
model.vision_encoder.freeze()
```

## Architecture Details

### Abstraction Layer

The multi-encoder support is implemented through:

1. **VisionEncoderBase**: Abstract base class defining the interface
2. **Registry Pattern**: Encoder types are registered and managed centrally
3. **Factory Method**: `create_vision_encoder()` creates the appropriate encoder

### Key Components

```
models/
├── vision_encoder_base.py      # Abstract interface
├── vision_encoder_registry.py   # Registry and factory
├── encoders/
│   ├── siglip_encoder.py       # SigLIP wrapper
│   └── dinov3_encoder.py       # DINOv3 wrapper
```

### Encoder Output Format

All encoders return a `VisionEncoderOutput` with:
- `features`: Patch embeddings [batch_size, num_patches, hidden_dim]
- `pooled_output`: Optional CLS token output (None for SigLIP)
- `num_patches`: Number of patch tokens
- `grid_shape`: (height, width) of the patch grid

## Preprocessing Differences

### SigLIP
- No normalization (simplified pipeline)
- Dynamic resize to ensure divisibility by patch size
- Standard tensor conversion

### DINOv3
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Specific preprocessing order: rescale → resize → normalize
- Dynamic resize with 518×518 default

## Performance Comparison

Based on training experiments:

| Encoder | Parameters | Frozen | Loss (50 steps) | Grad Norm | Throughput |
|---------|------------|--------|-----------------|-----------|------------|
| SigLIP | 86M | No | 5.6 → 1.3-4.0 | 15-300 | 6-10k tok/s |
| DINOv3-small | 30M | Yes | 7.2 → 1.2-4.0 | 10-20 | 9-14k tok/s |
| DINOv3-base | 86M | Yes | 7.6 → 1.4-5.0 | 10-50 | 8-11k tok/s |

## Adding New Encoders

To add a new vision encoder:

1. Create a new encoder class inheriting from `VisionEncoderBase`
2. Implement required methods: `forward()`, `from_pretrained()`, etc.
3. Register the encoder using the `@register_encoder` decorator
4. Add configuration to `VISION_ENCODER_CONFIGS` in the registry

Example:
```python
from models.vision_encoder_base import VisionEncoderBase
from models.vision_encoder_registry import register_encoder

@register_encoder("my_encoder")
class MyEncoder(VisionEncoderBase):
    def forward(self, images):
        # Your implementation
        pass

    @classmethod
    def from_pretrained(cls, cfg):
        # Load pretrained weights
        pass
```

## Technical Notes

### Modality Projector Compatibility

The modality projector automatically adapts to different encoder output dimensions:
- 384-dim (DINOv3-small) → 960-dim language space
- 768-dim (SigLIP, DINOv3-base) → 960-dim language space

### Memory Requirements

With frozen encoders:
- Reduced memory for gradients
- Faster backward pass
- Can use larger batch sizes

### CLS Token Handling

- **SigLIP**: No CLS token, uses all patch embeddings
- **DINOv3**: Has CLS token, but we extract only patch tokens for the VLM

### Register Tokens

DINOv3's register tokens are special learnable tokens that:
- Provide additional representational capacity
- Stabilize training
- Are excluded from the patch features passed to the VLM

## Troubleshooting

### Issue: IndexError with frozen encoder
**Solution**: The fix has been applied - optimizer learning rate updates now check if encoder is frozen

### Issue: Shape mismatch in token replacement
**Solution**: Ensure image tokens are present in input_ids when using forward pass

### Issue: Different loss scales between encoders
**Normal**: Different encoders may start with different initial loss values; focus on the trend

## Future Work

The abstraction layer is designed to easily support:
- CLIP variants
- EVA-CLIP
- SigLIP at different scales
- Custom vision encoders

Simply follow the pattern established for DINOv3 to add new encoders.