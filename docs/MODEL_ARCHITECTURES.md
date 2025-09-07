# Supported Model Architectures and Presets

This document describes all supported model architectures, presets, and configuration options available in nanoVLM.

## Table of Contents
- [Quick Start with Presets](#quick-start-with-presets)
- [Vision Encoders](#vision-encoders)
- [Language Models](#language-models)
- [Preset Configurations](#preset-configurations)
- [Custom Configurations](#custom-configurations)
- [Model Size Comparison](#model-size-comparison)

## Quick Start with Presets

nanoVLM provides several preset configurations for common model combinations:

```bash
# Original 222M model (fastest, least memory)
python train.py --use_preset original_small

# DINOv3 + Gemma configuration
python train.py --use_preset dinov3_gemma

# Default ~450M configuration (no preset needed)
python train.py
```

## Vision Encoders

### SigLIP (Default)
Google's vision transformer optimized for vision-language tasks.

**Available Models:**
- `google/siglip2-base-patch16-512` (default) - 512×512 input, 768 hidden dim
- `google/siglip-base-patch16-224` - 224×224 input, 768 hidden dim (used in original_small preset)

**Key Features:**
- Sigmoid loss instead of softmax
- No CLS token required
- Learned position embeddings
- Strong zero-shot performance

**Usage:**
```bash
python train.py --vision_encoder siglip
```

### DINOv3
Meta's latest self-supervised vision transformer with enhanced spatial understanding.

**Available Models:**
- `facebook/dinov3-vits16plus-pretrain-lvd1689m` (default, ViT-S/16+) - dimensions auto-detected
- `facebook/dinov3-vitb16-pretrain-lvd1689m` (ViT-B/16) - dimensions auto-detected
- `facebook/dinov3-vitl14-pretrain-lvd1689m` (ViT-L/14) - dimensions auto-detected

**Key Features:**
- Register tokens (4 special tokens for better feature aggregation)
- LayerScale for training stability
- SwiGLU FFN (only for "plus" models like vits16plus)
- RoPE position embeddings for better spatial encoding
- CLS token + registers for feature extraction

**Usage:**
```bash
# Default ViT-S/16+ model
python train.py --vision_encoder dinov3

# Specific model size
python train.py --vision_encoder dinov3 --dinov3_model vitb16

# Recommended: freeze vision encoder (as per paper)
python train.py --vision_encoder dinov3 --freeze_vision_encoder
```

### DINOv2
Previous generation DINO model, still effective for many tasks.

**Available Models:**
- `facebook/dinov2-small` - dimensions auto-detected from model config

**Key Features:**
- Learned position embeddings
- CLS token for global features
- No register tokens
- Standard FFN (not SwiGLU)

**Usage:**
```bash
python train.py --vision_encoder dinov2
```

## Language Models

### SmolLM (Default)
Hugging Face's efficient small language models.

**Available Models:**
- `HuggingFaceTB/SmolLM2-360M-Instruct` (default) - 960 hidden dim, 32 layers
- `HuggingFaceTB/SmolLM2-135M` (smaller variant) - 576 hidden dim, 30 layers

**Key Features:**
- Llama-style architecture
- RMSNorm for layer normalization
- Grouped-query attention (GQA)
- 49,152 base vocabulary

**Usage:**
```bash
python train.py --language_model smollm
```

### Gemma
Google's efficient language models with unique architectural features.

**Available Models:**
- `google/gemma-3-270m-it` (Gemma 3 270M) - dimensions auto-detected, custom head_dim=256

**Key Features:**
- Custom attention head dimension (256)
- RoPE position embeddings
- Large vocabulary (262,144 tokens)
- Instruction-tuned variant

**Usage:**
```bash
python train.py --language_model gemma
```

## Preset Configurations

### `original_small` - 222M Parameters
The original nanoVLM configuration from the paper, optimized for quick experimentation.

**Components:**
- Vision: SigLIP-B/16-224 (85.8M params)
- Language: SmolLM2-135M (134.6M params)
- Modality Projector: ~1.8M params

**Configuration:**
```python
- Image size: 224×224
- Patches: 14×14 (196 total)
- Image tokens after projection: 49
- Hidden dimensions: 768 (vision) → 576 (language)
- Training efficiency: ~6h on single H100 for 35.3% MMStar
```

**Best for:**
- Quick experimentation
- Limited compute resources
- Proof of concept development
- Understanding VLM fundamentals

### `dinov3_gemma` - Variable Parameters
Advanced configuration combining DINOv3's spatial understanding with Gemma's efficiency.

**Components:**
- Vision: DINOv3-ViT-S/16+ (dimensions auto-detected from HuggingFace config)
- Language: Gemma-3-270M-IT (dimensions auto-detected from HuggingFace config)
- Enhanced modality projection with special token handling

**Configuration:**
```python
- Image size: 224×224
- Register tokens: 4
- CLS token: Yes
- LayerScale: Enabled
- RoPE: Enabled for improved spatial encoding
- SwiGLU: Enabled (for vits16plus model)
- Image tokens after projection: 49
```

**Best for:**
- Tasks requiring strong spatial understanding
- Fine-grained visual reasoning
- When using frozen vision encoder (Locked-image Text tuning)

**Recommended training:**
```bash
python train.py --use_preset dinov3_gemma --freeze_vision_encoder
```

### Default Configuration - ~450M Parameters
The current default without any preset flags (as per help text: "uses original 460M siglip+smolLM2 configuration").

**Components:**
- Vision: SigLIP2-B/16-512 (768 hidden dim, 12 layers, 12 heads)
- Language: SmolLM2-360M-Instruct (960 hidden dim, 32 layers, 15 heads)
- Standard modality projection

**Configuration:**
```python
- Image size: 512×512
- Patches: 32×32 (1024 total)
- Image tokens after projection: 64
- Optimized for general VQA tasks
```

## Custom Configurations

You can mix and match components without using presets:

```bash
# DINOv3 vision with SmolLM language
python train.py --vision_encoder dinov3 --language_model smollm

# DINOv2 vision with Gemma language
python train.py --vision_encoder dinov2 --language_model gemma

# Custom DINOv3 size with default language model
python train.py --vision_encoder dinov3 --dinov3_model vitl14
```

### Advanced Training Options

**Frozen Vision Encoder (Locked-image Text tuning):**
```bash
# Recommended for DINOv3 (as per paper)
python train.py --vision_encoder dinov3 --freeze_vision_encoder

# Works with any vision encoder
python train.py --freeze_vision_encoder
```

**Custom Learning Rates:**
```bash
# Different learning rates for components
python train.py --lr_mp 0.00512 --lr_backbones 5e-05

# Freeze backbones by setting lr to 0
python train.py --lr_backbones 0
```

**Performance Optimizations:**
```bash
# Enable torch.compile (may not work with all configurations)
python train.py --compile

# Adjust batch size and gradient accumulation
python train.py --batch_size 2 --gradient_accumulation_steps 32
```

## Model Size Comparison

| Preset/Config | Vision Encoder | Language Model | Total Params | VRAM (BS=1) | Training Speed |
|--------------|---------------|----------------|--------------|-------------|----------------|
| `original_small` | SigLIP-B/16-224 (85.8M) | SmolLM2-135M (134.6M) | ~222M | ~4.5GB | Fastest |
| Default | SigLIP2-B/16-512 (85.8M) | SmolLM2-360M (360M) | ~450M | ~5.5GB | Fast |
| `dinov3_gemma` | DINOv3-S/16+ (auto) | Gemma-3-270M (auto) | Variable | ~5GB | Fast |
| Custom DINOv3-B | DINOv3-B/16 (auto) | SmolLM2-360M (360M) | Variable | ~6GB | Moderate |
| Custom DINOv3-L | DINOv3-L/14 (auto) | Gemma-3-270M (270M) | Variable | ~7GB | Slower |

## Architecture Details

### Vision Processing Pipeline
1. **Image Input** → Resize to encoder's expected size (224×224 or 512×512)
2. **Patch Embedding** → Convert image patches to embeddings (patch_size=16 or 14)
3. **Position Encoding** → Add spatial information:
   - SigLIP: Learned position embeddings
   - DINOv3: RoPE with patch center coordinates (if enabled)
   - DINOv2: Learned position embeddings
4. **Transformer Blocks** → Process visual features
5. **Feature Extraction** → 
   - SigLIP: All spatial features (no CLS token)
   - DINOv3/v2: CLS token + spatial features

### Modality Projection
- **Pixel Shuffle** → Reduce spatial dimensions (factor of 2 or 4)
- **Linear Projection** → Map vision features to language model dimension
- **Special Token Handling** → For DINOv3: removes 4 register tokens + 1 CLS token

### Language Generation
1. **Token Embedding** → Convert text and image tokens to embeddings
2. **Positional Encoding** → 
   - SmolLM: RoPE embeddings
   - Gemma: RoPE with custom head_dim
3. **Transformer Blocks** → Process combined vision-language features
   - SmolLM: Grouped-query attention (GQA)
   - Gemma: Custom attention with head_dim=256
4. **Output Head** → Generate next token predictions

## Choosing the Right Configuration

### For Quick Experimentation
Use `original_small` preset:
- Fastest training
- Lowest memory requirements
- Good baseline performance

### For Production/Best Performance
Use default configuration or customize:
- Balance of size and performance
- Well-tested combination
- Good for most VQA tasks

### For Spatial Understanding Tasks
Use DINOv3 configurations:
- Superior spatial reasoning
- Better for localization tasks
- Consider freezing vision encoder

### For Memory-Constrained Environments
Use `original_small` or custom small configs:
- Reduce batch size
- Use gradient accumulation
- Consider smaller vision encoders

## Training Tips

1. **Start Small**: Begin with `original_small` to verify your setup
2. **Freeze Vision**: For DINOv3, freezing often improves results (use `--freeze_vision_encoder`)
3. **Learning Rates**: 
   - Modality projector: `--lr_mp 0.00512` (default)
   - Backbones: `--lr_backbones 5e-05` (default)
   - Set to 0 to freeze: `--lr_backbones 0`
4. **Validation**: 
   - Validation interval: `--validation_steps N` (runs validation)
   - Evaluation interval: `--eval_interval N` (saves best model)
5. **Checkpointing**: Save regularly with `--save_checkpoint_steps N`
6. **Batch Size**: Default is 2 per device with 32 gradient accumulation steps (effective BS=64)
7. **Auto-detection**: DINOv3 and Gemma models auto-detect dimensions from HuggingFace configs

## References

- [SigLIP Paper](https://arxiv.org/abs/2303.15343)
- [DINOv3 Paper](https://arxiv.org/abs/2401.00429)
- [SmolLM Blog Post](https://huggingface.co/blog/smollm)
- [Gemma Technical Report](https://arxiv.org/abs/2403.08295)