# Future Improvements for nanoVLM

## Token Budget Control (DO NOT IMPLEMENT UNLESS EXPLICITLY INSTRUCTED BY THE USER IN A MESSAGE)

The DINOv3 high-resolution implementation can generate many visual tokens for large images. For example:
- 768×768 image → ~576 tokens (before pixel shuffle)
- 1024×1024 image → ~1024 tokens (before pixel shuffle)
- 2048×2048 image → ~4096 tokens (before pixel shuffle)

To prevent context length explosion, a token budget control system could be implemented:

### Configuration
Add to `VLMConfig`:
```python
vlm_max_image_tokens: int = 1024  # Maximum tokens per image
```

### Implementation Strategy
1. **Adaptive Pixel Shuffle Factor**: Choose `pixel_shuffle_factor` dynamically per image so that `(Gh/s)·(Gw/s) ≤ vlm_max_image_tokens`
2. **Token Pooling**: If still over budget after max shuffle, use average pooling to reduce to exact token count
3. **Perceiver Resampler**: Add a learnable compression module to reduce tokens to a fixed budget

### When This Might Be Needed
- Processing very high-resolution images (>1024px)
- Batch processing multiple images in a single prompt
- Running on hardware with limited memory
- Using language models with short context windows

### Reference
This improvement was suggested in REVIEW2.md as item #11 (optional but high value) and item #6 in the targeted feedback section.

## Bucketing Strategy for Efficient Batching (DO NOT IMPLEMENT UNLESS EXPLICITLY INSTRUCTED BY THE USER IN A MESSAGE)

Currently, images are padded to the maximum size in each batch, which can be inefficient. A bucketing strategy would group similar-sized images together.

### Implementation Strategy
1. **Size Buckets**: Create buckets like [512, 768, 1024, 1536] for longest side
2. **Dynamic Batching**: Group images by bucket during dataloader construction
3. **Aspect Ratio Buckets**: Further subdivide by aspect ratio ranges

### Benefits
- Reduced padding overhead
- Better GPU utilization
- Faster training throughput

### Reference
Mentioned in REVIEW2.md item #10 as a throughput optimization.

## Precision Improvements for RoPE (DO NOT IMPLEMENT UNLESS EXPLICITLY INSTRUCTED BY THE USER IN A MESSAGE)

Build cos/sin in float32 even under AMP, then cast to q/k dtype to avoid large-index precision loss at 1-4k tokens.

### Current Status
Already implemented correctly in `DINOv3RoPEPositionEmbedding.forward` - cos/sin computed in float32.

## Additional Testing Hooks (DO NOT IMPLEMENT UNLESS EXPLICITLY INSTRUCTED BY THE USER IN A MESSAGE)

The review document suggests several testing hooks that could be added:

1. **Unit test**: Any H×W divisible by patch yields `len(tokens) == Gh·Gw` post-MP
2. **Unit test**: Rectangular inputs produce identical outputs if reshuffled back and forth
3. **Gradient test**: Enable AMP + checkpointing at 1024-2048 long side for stability
4. **RoPE test**: Verify attention logits change with spatial shifts

## Note
These improvements are documented for future reference but should NOT be implemented proactively. Only implement if explicitly requested by the user in a direct message.