# Unified Architecture Design for SIGLIP + DINOv3

## Core Problem
SIGLIP and DINOv3 have fundamentally different image processing approaches:
- **SIGLIP**: Splits images into fixed 224x224 tiles, each producing 49 tokens
- **DINOv3**: Aspect-preserving resize, variable token count per image

## Key Insight
The difference is NOT in the vision encoder or modality projector - it's in:
1. **Image preparation** (split vs resize)
2. **Token counting** (fixed vs dynamic)
3. **Grid representation** (tiles vs patches)

## Unified Design

### 1. Data Processing Layer
```python
def get_image_processor(...):
    if single_image_mode:  # DINOv3
        # Returns: (tensor, {"HpWp": (Hp, Wp), "GhGw": (Gh, Gw)})
        return DINOv3Processor(...)
    else:  # SIGLIP
        # Returns: (tiles, (n_h, n_w))
        return SIGLIPProcessor(...)
```

### 2. VisionLanguageModel Forward Pass
```python
def forward(self, input_ids, images, attention_mask=None, targets=None, image_grids=None):
    # Detect processing mode from image_grids format
    is_dinov3 = (image_grids and 
                 isinstance(image_grids[0], dict) and 
                 "HpWp" in image_grids[0])
    
    if is_dinov3:
        # DINOv3 path: Pad images, track original sizes
        images = self._prepare_dinov3_batch(images)
        image_embd = self._process_dinov3(images, image_grids)
    else:
        # SIGLIP path: Stack tiles directly
        images = torch.stack(images) if images[0].dim() == 3 else torch.cat(images)
        image_embd = self.vision_encoder(images)
        image_embd = self.MP(image_embd)
```

### 3. Modality Projector
```python
class ModalityProjector(nn.Module):
    def forward(self, x, gh=None, gw=None):
        # Remove special tokens if configured
        if self.handle_special_tokens:
            x = self._remove_special_tokens(x)
        
        # Apply appropriate pixel shuffle
        if gh is not None and gw is not None:
            x = self.pixel_shuffle_2d(x, gh, gw)  # DINOv3
        else:
            x = self.pixel_shuffle(x)  # SIGLIP
        
        return self.proj(x)
```

### 4. Token Generation
```python
def get_image_string(tokenizer, grid_info, mp_image_token_length):
    if mp_image_token_length == 1:  # DINOv3
        # Dynamic tokens based on grid
        n_h, n_w = grid_info["GhGw"]
        return tokenizer.image_token * (n_h * n_w)
    else:  # SIGLIP
        # Fixed tokens per tile with position markers
        n_h, n_w = grid_info
        tokens = ""
        for i in range(n_h):
            for j in range(n_w):
                tokens += f"r{i+1}c{j+1}" + tokenizer.image_token * mp_image_token_length
        return tokens
```

## Implementation Strategy

### Phase 1: Clean Separation
1. Keep SIGLIP and DINOv3 paths completely separate in VLM forward
2. Detect mode from config or data format
3. No shared processing between modes

### Phase 2: Test Both Modes
1. Test SIGLIP with original_small preset
2. Test DINOv3 with dinov3 config
3. Ensure both produce correct token counts

### Phase 3: Validation
1. Loss should decrease for both
2. No dimension mismatches
3. Clean console output

## Critical Rules
1. **Never mix tile and patch processing**
2. **Token count must match data preparation**
3. **Grid format determines processing path**
4. **Config drives behavior, not heuristics**

## Configuration Flags
```python
# SIGLIP
vit_architecture = "siglip"
mp_pixel_shuffle_factor = 2
mp_image_token_length = 49
mp_handle_special_tokens = False

# DINOv3
vit_architecture = "dinov3"
mp_pixel_shuffle_factor = 2
mp_image_token_length = 1
mp_handle_special_tokens = True
```