# DINOv3 High-Resolution Implementation Troubleshooting

This document details common issues and solutions when implementing DINOv3 with high-resolution image support in nanoVLM.

## Common Issues and Solutions

### 1. AssertionError in Pixel Shuffle

**Symptom**: Training fails with `AssertionError` in `pixel_shuffle` function, typically with message about sequence length not being square.

**Cause**: Image grids are not being properly propagated through the data pipeline, causing fallback to square-only pixel shuffle.

**Solution**: 
- Verify `image_grids` are returned from dataset and preserved in collator
- Ensure `_discard_samples_that_are_too_long` and `_truncate_samples_that_are_too_long` handle `image_grids`
- Check train.py extracts and passes `image_grids` to model.forward()

### 2. Tensor Size Mismatch During Concatenation

**Symptom**: `RuntimeError: Sizes of tensors must match except in dimension 0`

**Cause**: When processing variable-sized images, projected embeddings have different sequence lengths and cannot be concatenated directly.

**Solution**:
- Add padding before concatenation:
```python
max_seq_len = max(p.shape[1] for p in projected)
padded_projected = []
for p in projected:
    if p.shape[1] < max_seq_len:
        pad_size = max_seq_len - p.shape[1]
        padded = F.pad(p, (0, 0, 0, pad_size), "constant", 0)
        padded_projected.append(padded)
    else:
        padded_projected.append(p)
image_embd = torch.cat(padded_projected, dim=0)
```

### 3. Dummy Batch Grid Dimension Errors

**Symptom**: `AssertionError: Grid dimensions (gh=1, gw=1) must be divisible by scale factor 2`

**Cause**: Dummy batch created when all samples are filtered out uses invalid grid dimensions.

**Solution**:
- Ensure dummy batch uses grid dimensions divisible by pixel_shuffle_factor:
```python
return {
    "input_ids": dummy_tensor,
    "attention_mask": dummy_tensor,
    "images": [dummy_image],
    "labels": dummy_tensor,
    "image_grids": [(2, 2)],  # Must be divisible by pixel_shuffle_factor=2
}
```

### 4. Image Grids Lost in Data Pipeline

**Symptom**: `DEBUG: image_grids type: <class 'list'>, len: 0` even when images should have grids

**Cause**: Grid information is lost during data transformations in collator or dataset.

**Solution**:
- Update VQACollator to preserve `image_grids` through all transformations
- Verify `_discard_samples_that_are_too_long` and `_truncate_samples_that_are_too_long` handle image_grids
- Add debug prints at each stage to track grid propagation

### 5. Aspect Ratio Distortion

**Symptom**: Images appear stretched or compressed after processing

**Cause**: Images are being resized to square dimensions instead of preserving aspect ratio.

**Solution**:
- Use `DynamicResize` with `allow_upscale=False` for aspect-preserving resize
- Ensure processor calculates grid dimensions from actual (not padded) image sizes
- Verify padding is applied only during batching, not during initial resize

## Debugging Tips

### Add Temporary Debug Statements

During development, add debug prints to track grid propagation:

```python
# In vision_language_model.py
def forward(...):
    print(f"Vision encoder output shape: {image_embd.shape}")
    print(f"image_grids type: {type(image_grids)}, len: {len(image_grids) if image_grids is not None else 'N/A'}")
    
    if hasattr(self, "_batch_original_sizes"):
        print("Using _batch_original_sizes")
        for i, (orig_h, orig_w) in enumerate(self._batch_original_sizes):
            print(f"Image {i} - orig: {orig_h}x{orig_w}, grid: {Hp}x{Wp}")
```

### Verify Grid Calculations

Check that grid dimensions are calculated correctly:

```python
# In processors.py
Hp, Wp = H // vit_patch_size, W // vit_patch_size  # Patch grid
Gh, Gw = Hp // pixel_shuffle_factor, Wp // pixel_shuffle_factor  # Final grid
assert Hp % pixel_shuffle_factor == 0 and Wp % pixel_shuffle_factor == 0
```

## Best Practices

1. **Always test with rectangular images** - Don't just test with square images
2. **Verify grid propagation** - Add debug prints to ensure grids make it through entire pipeline
3. **Use dummy batches with valid dimensions** - Ensure dummy data has valid grid dimensions
4. **Pad before concatenation** - Always pad variable-length sequences before concatenation
5. **Remove debug statements before commit** - Replace prints with proper logging for production code

## Performance Considerations

- **Memory usage**: High-resolution images generate more tokens, monitor GPU memory
- **Batch size**: May need to reduce batch size for very high-resolution images
- **Padding overhead**: Consider implementing bucketing strategy for more efficient batching (see FUTURE_IMPROVEMENTS.md)

## Verification Checklist

Before committing DINOv3 high-resolution changes:

- [x] Image grids propagate through entire pipeline (dataset → collator → model)
- [x] Variable-sized images are handled correctly with proper padding
- [x] Dummy batches use valid grid dimensions
- [x] All debug statements are removed
- [x] Training runs successfully with rectangular images
- [x] Aspect ratio is preserved throughout processing
- [x] Commit message includes detailed description of changes