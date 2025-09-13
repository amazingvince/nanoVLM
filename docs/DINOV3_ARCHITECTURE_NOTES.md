# DINOv3 Architecture Analysis - Working Version 46a519c9

## Key Observations

### 1. Image Processing Pipeline  
- **Transform**: `DynamicResize` → `ToTensor` → `Normalize`
- **DynamicResize**: Aspect-preserving resize with dimensions divisible by `patch_size * pixel_shuffle_factor`
- Returns: (tensor, grid_info) where grid_info = {"HpWp": (Hp, Wp), "GhGw": (Gh, Gw)}

### 2. Vision Language Model Forward Pass
```python
def forward(self, input_ids, images, attention_mask=None, targets=None, image_grids=None):
    # Pad images to same size for batching
    if images[0].dim() == 3:
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        
        # Store original sizes
        original_sizes = [(img.shape[1], img.shape[2]) for img in images]
        
        # Pad each image
        padded_images = []
        for img in images:
            pad_h = max_h - img.shape[1]
            pad_w = max_w - img.shape[2]
            padded = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
            padded_images.append(padded)
        
        images = torch.stack(padded_images)
        self._batch_original_sizes = original_sizes
    
    # Process through vision encoder
    image_embd = self.vision_encoder(images)
    
    # Apply modality projector with original dimensions
    for i, (orig_h, orig_w) in enumerate(self._batch_original_sizes):
        Hp = orig_h // patch_size
        Wp = orig_w // patch_size
        
        img_features = image_embd[i:i+1]
        proj_embd = self.MP(img_features, gh=Hp, gw=Wp)
        projected.append(proj_embd)
```

### 3. Modality Projector
- Has `pixel_shuffle_2d` method for rectangular grids
- Handles padding by extracting only real tokens based on gh, gw
- `mp_pixel_shuffle_factor = 2` for DINOv3  
- `mp_image_token_length = 1` (1 token per grid cell)
- `mp_handle_special_tokens = True` (removes CLS + registers)

### 4. Token Generation
```python
def get_image_string(tokenizer, splitted_image_counts, mp_image_token_length):
    # For DINOv3 with mp_image_token_length=1:
    if mp_image_token_length == 1:
        # Emit exactly n_h * n_w tokens for the image
        image_string += tokenizer.image_token * (n_h * n_w)
```

### 5. Key Design Principles
- **Aspect-preserving**: Maintains image aspect ratio
- **Dynamic grids**: Variable number of tokens per image
- **Batch padding**: Pads images to same size for batching
- **Original size tracking**: Uses original dimensions for modality projector
- **Special token handling**: Removes CLS and register tokens before projection
- **RoPE embeddings**: Uses rotary position embeddings with actual patch dimensions

## Working Configuration  
- `vit_architecture = "dinov3"`
- `vit_img_size = 1024` (can handle high-res)
- `vit_patch_size = 16`
- `vit_cls_flag = True`
- `vit_num_registers = 4`
- `vit_use_rope = True`
- `mp_pixel_shuffle_factor = 2`
- `mp_image_token_length = 1`
- `mp_handle_special_tokens = True`

## Critical Success Factors
1. Images padded to same size for batch processing
2. Original sizes tracked for correct grid dimensions
3. Modality projector extracts only real (non-padded) tokens
4. Token count is dynamic based on actual image dimensions
5. RoPE embeddings use actual patch grid dimensions