# Data Preprocessing Pipeline Documentation

## Overview

This document provides a comprehensive developer-level overview of the data preprocessing and handling logic in nanoVLM. The pipeline is designed to prepare multimodal (image + text) data for training a Vision-Language Model that combines a SigLIP2 vision encoder with a SmolLM2 language decoder.

## Architecture Components

### 1. Vision Encoder (SigLIP2-specific)
- **Model**: `google/siglip2-base-patch16-512`
- **Input Requirements**: 
  - Image size: 512×512 pixels
  - Patch size: 16×16 pixels
  - Number of patches: (512/16)² = 1024 patches
  - No CLS token (SigLIP design choice)

### 2. Language Model 
- **Model**: `HuggingFaceTB/SmolLM2-360M-Instruct`
- **Tokenizer**: SmolLM2 tokenizer with 49,152 base tokens
- **Extended Vocabulary**: +66 special tokens for image placeholders

### 3. Modality Projection
- **Pixel Shuffle Factor**: 4
- **Output Token Length**: 64 tokens per image patch
- **Purpose**: Reduces spatial dimensions while increasing feature depth

## Data Flow During Training

### Step 1: Dataset Loading
```python
# From train.py -> get_dataloaders()
dataset = load_dataset(
    train_cfg.train_dataset_path,
    train_cfg.train_dataset_name,
    split="train"
)
```

### Step 2: Image Processing Pipeline

#### 2.1 Dynamic Resizing (`custom_transforms.py::DynamicResize`)

**Purpose**: Ensure images are compatible with patch-based vision transformer

**Logic**:
```python
1. Input: PIL Image or tensor of arbitrary size
2. Calculate target dimensions:
   - Longer side ≤ max_side_len (default: 1024px)
   - Both dimensions divisible by patch_size (16px for SigLIP2)
   - Maintains aspect ratio
3. Resize using BICUBIC interpolation
4. Output: Resized image with dimensions divisible by 16
```

**SigLIP2-specific**: The 16px patch size requirement comes directly from SigLIP2's architecture (`patch16` in model name).

#### 2.2 Tensor Conversion
```python
transforms.ToTensor()
# Converts PIL Image to tensor
# Normalizes from [0, 255] to [0, 1]
# Reorders from (H, W, C) to (C, H, W)
```

**Note**: Unlike standard SigLIP preprocessing, nanoVLM does NOT apply ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). This is intentional to simplify the pipeline.

#### 2.3 Image Splitting (`custom_transforms.py::GlobalAndSplitImages`)

**Purpose**: Enable high-resolution image understanding through patch-based processing

**Logic**:
```python
1. Input: Tensor of shape (C, H, W) where H, W divisible by patch_size
2. Split image into grid of patches:
   - Each patch: patch_size × patch_size (512×512 for full resolution)
   - Grid dimensions: (n_h, n_w) where n_h = H/patch_size, n_w = W/patch_size
3. Add global context patch:
   - Resize entire image to patch_size × patch_size
   - Prepend to patch list (unless only 1 patch exists)
4. Output: (patches, grid_dims)
   - patches: Tensor of shape (num_patches, C, patch_size, patch_size)
   - grid_dims: Tuple (n_h, n_w)
```

**Maximum splits**: Up to 8×8 grid = 64 patches + 1 global = 65 patches total

### Step 3: Text Processing Pipeline

#### 3.1 Message Construction (`datasets.py::_get_messages()`)

**Quality Filtering**:
- Filters based on rating thresholds:
  - `relevance_min_rating`
  - `image_correspondence_min_rating`  
  - `visual_dependency_min_rating`
  - `formatting_min_rating`
- Skips samples below thresholds

**Image Token Injection**:
```python
1. Create user/assistant message pairs
2. Generate image token string:
   - Global image token: "<|global_image|>" + 64 × "<|image|>"
   - For each patch position (row, col):
     - Position token: "<row_X_col_Y>" 
     - Image tokens: 64 × "<|image|>"
3. Prepend image tokens to first user message
```

#### 3.2 Tokenization (`datasets.py::_prepare_inputs_and_loss_mask()`)

```python
1. Apply chat template to messages
2. Tokenize with SmolLM2 tokenizer
3. Create loss mask:
   - 0 for user messages (ignored in loss)
   - 1 for assistant messages (included in loss)
   - Accounts for prefix length in chat template
```

### Step 4: Batch Collation

#### 4.1 Standard Collation (`collators.py::VQACollator`)

For regular training:
```python
1. Filter samples exceeding max_length
2. Pad sequences to uniform length:
   - input_ids: Pad with pad_token_id
   - labels: Pad with -100 (ignored in loss)
   - attention_mask: Pad with 0
3. Stack into batch tensors
```

#### 4.2 Constant Length Packing (`advanced_datasets.py::ConstantLengthDataset`)

For efficient training with packed sequences:
```python
1. Buffer samples until total_length > max_length
2. Apply knapsack algorithm:
   - Group samples to minimize padding
   - Respect max_images_per_knapsack constraint
   - Balance load across knapsacks
3. Pack groups into fixed-length sequences
4. Use producer-consumer threading for efficiency
```

### Step 5: Model Forward Pass

#### 5.1 Vision Encoding (`vision_transformer.py`)

**SigLIP2-specific processing**:
```python
1. Extract patches using Conv2d (kernel=16, stride=16)
2. Flatten patches: (B, C, n_h, n_w) -> (B, n_patches, hidden_dim)
3. Add learned position embeddings
4. Process through 12 transformer blocks
5. Apply LayerNorm (no CLS token pooling)
6. Output: (B, n_patches, 768)
```

#### 5.2 Modality Projection (`modality_projector.py`)

```python
1. Pixel shuffle to reduce spatial dimensions:
   - Input: (B, 1024, 768)  # 32×32 patches
   - Reshape to spatial: (B, 32, 32, 768)
   - Shuffle with factor 4: (B, 8, 8, 768×16)
   - Output: (B, 64, 12288)
2. Linear projection to language dimension:
   - Project: 12288 -> 960
   - Output: (B, 64, 960)
```

#### 5.3 Token Embedding Replacement (`vision_language_model.py`)

```python
1. Get text token embeddings from decoder
2. Find image token positions (token_id == image_token_id)
3. Replace image tokens with projected vision features
4. Maintain original sequence length and order
```

## Special Tokens Mapping

### Image Placeholder Tokens (66 total)
- `<|image|>`: Base image token placeholder
- `<|global_image|>`: Global context patch marker
- `<row_X_col_Y>`: Grid position markers (8×8 = 64 positions)

These tokens are added to the SmolLM2 vocabulary (49,152 → 49,218 tokens).

## Key Design Decisions

### 1. No ImageNet Normalization
Unlike standard vision models, nanoVLM skips mean/std normalization. This simplifies preprocessing but may affect transfer learning from pretrained weights.

### 2. Dynamic Resolution Support
Images can be processed at various resolutions (up to 2048×2048) through the grid splitting mechanism, enabling fine-grained visual understanding.

### 3. Global + Local Processing
The global context patch preserves whole-image understanding while local patches enable detailed analysis.

### 4. Efficient Packing
The ConstantLengthDataset with knapsack packing minimizes padding waste during training.

## Configuration Parameters

### Vision-specific (from `config.py::VLMConfig`)
- `vit_img_size`: 512 (SigLIP2 native resolution)
- `vit_patch_size`: 16 (SigLIP2 architecture)
- `mp_pixel_shuffle_factor`: 4 (compression ratio)
- `mp_image_token_length`: 64 (tokens per patch)
- `max_img_size`: 1024 (maximum input resolution)

### Training-specific (from `config.py::TrainConfig`)
- `max_sample_length`: 2048 (maximum sequence length)
- `max_images_per_example`: 8 (per-sample limit)
- `max_images_per_knapsack`: 36 (per-batch limit)

## Preprocessing Validation Checklist

When modifying the preprocessing pipeline, ensure:

1. ✅ Image dimensions remain divisible by `vit_patch_size` (16)
2. ✅ Number of image tokens matches vision encoder output
3. ✅ Special tokens are properly registered in tokenizer
4. ✅ Loss masks correctly exclude padding and user messages
5. ✅ Batch dimensions align across vision and language components
6. ✅ Memory constraints respected (image count limits)

## Common Issues and Solutions

### Issue: RuntimeError on position embedding mismatch
**Cause**: Image size not matching expected patch count
**Solution**: Ensure DynamicResize produces dimensions divisible by 16

### Issue: Shape mismatch in token replacement
**Cause**: Mismatch between image token count and vision features
**Solution**: Verify `mp_image_token_length` × patch_count alignment

### Issue: OOM during training
**Cause**: Too many high-resolution images in batch
**Solution**: Reduce `max_images_per_knapsack` or `max_img_size`

## Future Improvements

1. **Add SigLIP normalization**: Include proper mean/std normalization for better transfer learning
2. **Dynamic token allocation**: Vary tokens per patch based on image complexity
3. **Adaptive resolution**: Automatically select optimal resolution per image
4. **Compression optimization**: Explore learned compression in modality projector