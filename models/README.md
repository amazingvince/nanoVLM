# Models

## Vision Backbone (ViT)

This is a very lightweight Vision Transformer in native pytorch with support for multiple architectures including SigLIP and DINOv3. I took inspiration from the following sources:
- https://github.com/karpathy/nanoGPT (General Transformer Decoder)
- https://arxiv.org/abs/2010.11929 (ViT Paper)
- https://arxiv.org/abs/2303.15343 (SigLiP Paper)
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py (HF SigLiP Implementation)
- https://arxiv.org/abs/2409.17826 (DINOv3 Paper)

### DINOv3 Support

The vision transformer now includes DINOv3 support with:
- **Rotary Position Embeddings (RoPE)**: Custom implementation of 2D RoPE for vision transformers following DINOv3's approach
- **Patch-based processing**: Native support for DINOv3's patch embedding and attention mechanisms
- **Pretrained weights loading**: Compatible with Facebook's DINOv3 pretrained models (e.g., `facebook/dinov3-vits16plus-pretrain-lvd1689m`)
- **Configurable RoPE parameters**: Support for custom base frequency (theta) for position embeddings

## Language Model (Llama / SmolLM / Gemma)

This is a decoder only LM, supporting multiple architectures including Llama 2/3, SmolLM, and Gemma models. Inspiration from the following sources:
- https://arxiv.org/pdf/2307.09288 (Original Llama Paper)
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py (HF Llama Implementation)
- https://arxiv.org/abs/2403.08295 (Gemma Paper)

### Gemma-3 Support

The language model now includes Gemma-3 support with:
- **Custom head dimensions**: Support for Gemma-3's larger attention head dimensions (256 instead of the typical hidden_dim/n_heads)
- **Compatible tokenizer**: Works with Gemma's tokenizer and special tokens
- **Instruction-tuned models**: Support for instruction-tuned variants like `google/gemma-3-270m-it`
- **Flexible architecture**: Automatic configuration loading from HuggingFace model configs

## Modality Projection

This is a simple MLP (Linear Layer) for the Modality Projection between the Image Patch Encodings and the Language Embedding Space with a simple Pixel Shuffle (https://arxiv.org/pdf/2504.05299)

## Vision-Language-Model

This brings all the individual parts together and handles the concatenation of images and text. Built as a simple version of SmolVLM (https://arxiv.org/pdf/2504.05299)

### Supported Configurations

The VLM supports various combinations of vision and language models:
- **SigLIP + Llama/SmolLM**: Original configuration following SmolVLM architecture
- **DINOv3 + Gemma-3**: New configuration combining DINOv3 vision encoder with Gemma-3 language decoder
- **Mixed configurations**: Flexible architecture allows mixing different vision encoders with language decoders

### Usage Example

```python
from models.config import get_dinov3_gemma_config

# Get a pre-configured DINOv3 + Gemma-3 setup
cfg = get_dinov3_gemma_config()

# The configuration includes:
# - DINOv3 vision encoder (facebook/dinov3-vits16plus-pretrain-lvd1689m)
# - Gemma-3 270M instruction-tuned model (google/gemma-3-270m-it)
# - Optimized training parameters for this combination
```
