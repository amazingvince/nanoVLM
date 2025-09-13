# SIGLIP Architecture Analysis - Working Version 74e50abe

## Key Observations

### 1. Image Processing Pipeline
- **Transform**: `DynamicResize` → `ToTensor` → `SplitImage`
- **SplitImage**: Splits large images into 224x224 tiles
- Returns: (patches, grid) where grid is (n_h, n_w) - number of tiles

### 2. Vision Language Model Forward Pass
```python
def forward(self, input_ids, images, attention_mask=None, targets=None):
    # Images come as list of tiles
    if isinstance(images[0], list):
        images = [img for sublist in images for img in sublist]
    
    # Stack tiles: [N, 3, H, W]
    if images[0].dim() == 3:
        images = torch.stack(images, dim=0)
    else:
        images = torch.cat(images, dim=0)
    
    # Process through vision encoder (all tiles at once)
    image_embd = self.vision_encoder(images)  # [N_tiles, seq_len, hidden_dim]
    
    # Apply modality projector
    image_embd = self.MP(image_embd)  # [N_tiles, mp_image_token_length, D_lm]
```

### 3. Modality Projector
- Uses `pixel_shuffle` to reduce tokens from 196 (14x14) to 49 (7x7) per tile
- `mp_pixel_shuffle_factor = 2` for SIGLIP
- `mp_image_token_length = 49`

### 4. Token Generation
```python
def get_image_string(tokenizer, splitted_image_counts, mp_image_token_length):
    # For each image's grid (n_h, n_w):
    for i in range(n_h):
        for j in range(n_w):
            grid_token_name = f"r{i + 1}c{j + 1}"
            image_string += getattr(tokenizer, grid_token_name)
            image_string += tokenizer.image_token * mp_image_token_length
```

### 5. Key Design Principles
- **Tile-based**: Splits images into fixed 224x224 tiles
- **Batch processing**: All tiles processed together through vision encoder
- **Fixed token count**: Each tile produces exactly 49 tokens
- **Grid tokens**: Special tokens mark tile positions (r1c1, r1c2, etc.)

## Working Configuration
- `vit_img_size = 224`
- `vit_patch_size = 16`
- `mp_pixel_shuffle_factor = 2`
- `mp_image_token_length = 49`
- `max_img_size = 224` (for original_small preset)

## Critical Success Factors
1. All tiles are same size (224x224)
2. Each tile produces fixed number of patches (14x14 = 196)
3. Pixel shuffle reduces to fixed token count (7x7 = 49)
4. Token count is deterministic based on grid size