**Purpose**
Enable true high-resolution DINOv3 in nanoVLM. We remove the 224×224 clamp, turn on RoPE as intended, and make the visual token pipeline respect rectangular grids. This improves OCR/text, documents, UI screenshots, and any task that benefits from >224 inputs.

**Why**

* DINOv3 trains and adapts at higher resolutions and relies on RoPE to stay consistent across H×W.
* Our code path resized to 224 and assumed square feature maps, throwing away detail and blocking HR inference.
* The ModalityProjector and data pipeline hid the real grid size, so token counts and positions were wrong for aspect-preserving inputs.

**What changes**

* **RoPE enabled for DINOv3** and applied to patch tokens using the **true** patch grid `(Hp,Wp)` from the conv embed. No `sqrt(N)` guesses.
* **Image processor** now preserves aspect ratio, snaps both sides to **multiples of patch size**, adds explicit **Normalize(mean,std)**, and returns `(tensor, (Gh,Gw))` after the planned pixel-shuffle factor. No forced 224.
* **ModalityProjector** accepts **rectangular grids** and performs 2D pixel-shuffle with `(Gh,Gw)`. Square assert removed.
* **Grid plumbing**: `(Gh,Gw)` is threaded from the processor → encoder → MP.
* **Token placeholders**: dataset/text construction now emits exactly `Gh×Gw` image tokens; no fixed “49 per image.”
* **Label padding**: collators pad labels with `-100` (loss mask), not `pad_token_id`.
* **Optional controls**: config hooks for `max_img_size`, `pixel_shuffle_factor`, and a future `vlm_max_image_tokens` cap.

**Behavioral impact**

* Models consume variable-length visual token grids that match the resized image.
* Rectangular images work end-to-end.
* Performance should improve on text-heavy inputs and tall/wide layouts.
* Memory scales with tokens; long-side 768 produces \~12× the tokens of 224 at /16 before downsampling. Batch sizes may need tuning.

**Backwards compatibility**

* Processor now returns a **tuple** `(tensor, (Gh,Gw))`. Call sites that assumed a plain tensor were updated.
* `ModalityProjector.forward` now accepts `gh, gw`. Legacy path still works if not provided, but DINOv3 callers should pass them.
* Config: set `vit_use_rope=True` for DINOv3 and use `mp_image_token_length=1` (or remove the multiplier usage).

**Rollout plan**

1. Land ViT + RoPE grid fix.
2. Land ModalityProjector 2D + API change.
3. Land processor changes + normalization.
4. Wire `(Gh,Gw)` through VLM and scripts.
5. Remove fixed token multiplier and update collators.
6. Tune `max_img_size` and `pixel_shuffle_factor`; adjust batch sizes.

**Risk & mitigation**

* **OOM from token growth**: start with `max_img_size=768`, `pixel_shuffle_factor=4`. Add a token cap in a follow-up.
* **Throughput regression**: use dataloader **size bucketing**.
* **RoPE precision at long contexts**: compute cos/sin in fp32, then cast.
* **Checkpoint normalization mismatch**: we standardize on ImageNet mean/std. If you load a checkpoint with different stats, override in config.

**How to use**

```yaml
vit_architecture: dinov3
vit_patch_size: 16
vit_use_rope: true
max_img_size: 768          # cap longest side, aspect preserved
mp_pixel_shuffle_factor: 4 # downsample grid before LM
mp_handle_special_tokens: true
# Optional later:
# vlm_max_image_tokens: 1024
```

**Test plan**

* Unit: 640×960 input → expect Hp=40,Wp=60; with s=4 → Gh=10,Gw=15; MP output length = 150.
* Regression: compare 224 vs 768 on a small OCR set; expect accuracy lift.
* Stability: AMP training at 768 long-side; no NaNs; memory within budget.
* Sanity: rectangular images no longer trigger square asserts; inference scripts unpack `(tensor, grid)` and run.

**Summary**
We align nanoVLM with DINOv3’s intended HR+RoPE design, remove square-only assumptions, and make token counts faithful to image geometry. This unlocks higher-resolution use cases with controlled memory growth.


---

## detailed

Yes. Your plan is correct. Below is a concrete code review with precise locations, why it matters, recommended fixes, and an execution order. I include minimal diffs you can paste.

# Priority order

1. Enable DINOv3 RoPE and correct grid handling in the ViT.
2. Make ModalityProjector work on rectangular grids.
3. Fix the image processor to stop forcing 224 and to return true grids. Add normalization.
4. Wire `(Gh,Gw)` through your pipeline.
5. Remove the hard “49 tokens per image” assumption.
6. Fix inference scripts that treat the processor output as a tensor.
7. Clean up label padding.
8. Add token budget control and dataloader bucketing.

---

# 1) Vision encoder: enable RoPE and real H×W

## 1a. Wrong DINOv3 positional setting and disabled RoPE

**File**: `models/vision_transformer.py`
**Lines**: \~447–470

Issue: you set

```py
# 467-469
cfg.vit_use_rope = False
cfg.vit_use_sincos_pos = True
```

and the comment claims DINOv3 uses sin/cos instead of RoPE. This is incorrect for the DINOv3 variant you’re targeting.

**Fix**:

* For DINOv3: `vit_use_rope=True`, `vit_use_sincos_pos=False`.
* Keep sin/cos path only for SigLIP or if you explicitly choose it.

**Patch**:

```diff
@@
-            # DINOv3 uses sin/cos embeddings, not RoPE or learned embeddings
-            cfg.vit_use_rope = False
-            cfg.vit_use_sincos_pos = True  # DINOv3 uses sin/cos position embeddings
+            # DINOv3 uses RoPE with patch-center coordinates
+            cfg.vit_use_rope = True
+            cfg.vit_use_sincos_pos = False
```

## 1b. RoPE gate never triggers in attention

**File**: `models/vision_transformer.py`
**Lines**: ViTMultiHeadAttention `__init__` and `forward` (\~212–320)

Issue: `forward()` checks `getattr(self, "use_dinov3_rope", False)` but you never set `self.use_dinov3_rope`.

**Fix**: define it from config in `__init__`, or just key off the presence of `position_embeddings`.

**Patch (simplest)**:

```diff
@@ class ViTMultiHeadAttention(nn.Module):
-        self.use_rope = cfg.vit_use_rope if hasattr(cfg, "vit_use_rope") else False
+        self.use_rope = getattr(cfg, "vit_use_rope", False)
+        self.use_dinov3_rope = self.use_rope and getattr(cfg, "vit_architecture", "siglip") == "dinov3"
@@ def forward(self, x, position_embeddings=None):
-        if getattr(self, "use_dinov3_rope", False) and position_embeddings is not None:
+        if self.use_dinov3_rope and position_embeddings is not None:
```

## 1c. Using `sqrt(N)` breaks rectangular inputs

**File**: `models/vision_transformer.py`
**Lines**: ViT.forward \~404–420

Issue: you estimate `num_patches_h = num_patches_w = int(sqrt(...))`. This fails when the input preserves aspect ratio.

**Fix**: track H\_p and W\_p from the patch embed.

**Patch**:

1. Record the grid inside `ViTPatchEmbeddings.forward`:

```diff
@@ class ViTPatchEmbeddings(nn.Module):
-    def forward(self, x):
+    def forward(self, x):
         B = x.shape[0]
-        x = self.conv(x)
+        x = self.conv(x)  # B, C, Hp, Wp
+        Hp, Wp = x.shape[-2], x.shape[-1]
+        self._last_hw = (Hp, Wp)
         x = x.flatten(2).transpose(1, 2)
         ...
         return x
```

2. Use the stored grid in `ViT.forward`:

```diff
@@ class ViT(nn.Module):
-        if use_dinov3_rope and hasattr(self, "rope_embeddings"):
-            num_patches_h = num_patches_w = int(
-                math.sqrt(
-                    (x.shape[1] - 1 - getattr(self.cfg, "vit_num_registers", 0))
-                    if self.cls_flag
-                    else x.shape[1] - getattr(self.cfg, "vit_num_registers", 0)
-                )
-            )
+        if use_dinov3_rope and hasattr(self, "rope_embeddings"):
+            # Pull Hp,Wp from patch embed computed on this input
+            num_patches_h, num_patches_w = getattr(self.patch_embedding, "_last_hw")
             cos, sin = self.rope_embeddings(num_patches_h, num_patches_w, x.dtype, x.device)
             cos = cos.unsqueeze(0).unsqueeze(0)
             sin = sin.unsqueeze(0).unsqueeze(0)
             position_embeddings = (cos, sin)
```

---

# 2) ModalityProjector: support rectangular grids

**File**: `models/modality_projector.py`
**Lines**: \~39–60 and `forward`

Issue: `pixel_shuffle` assumes perfect squares:

```py
assert seq_root**2 == seq
height = width = seq_root
```

This rejects aspect-preserving inputs.

**Fix**: add a 2D path and let `forward` accept `gh, gw`.

**Patch**:

```diff
@@ class ModalityProjector(nn.Module):
-    def pixel_shuffle(self, x):
+    def pixel_shuffle(self, x):
         ...
         return x

+    def pixel_shuffle_2d(self, x, gh: int, gw: int):
+        bsz, seq, embed_dim = x.size()
+        assert seq == gh * gw, f"seq {seq} != gh*gw {gh*gw}"
+        s = self.scale_factor
+        assert gh % s == 0 and gw % s == 0
+        ho, wo = gh // s, gw // s
+        x = x.view(bsz, gh, gw, embed_dim)
+        x = x.view(bsz, ho, s, wo, s, embed_dim).permute(0,1,3,2,4,5).contiguous()
+        x = x.view(bsz, ho*wo, embed_dim * s * s)
+        return x
@@
-    def forward(self, x):
+    def forward(self, x, gh: int | None = None, gw: int | None = None):
         ...
-        x = self.pixel_shuffle(x)
+        if gh is not None and gw is not None:
+            x = self.pixel_shuffle_2d(x, gh, gw)
+        else:
+            x = self.pixel_shuffle(x)
         x = self.proj(x)
         return x
```

---

# 3) Image processor: stop forcing 224. Return true grids. Add normalization.

**File**: `data/processors.py`
**Lines**: \~63–78

Issues:

* `single_image_mode` resizes to a square and returns a fake `(1,1)` grid.
* No normalization.

**Fix**: use `DynamicResize` to max side `max_img_size`, snap to multiples of `vit_patch_size`, preserve aspect. Compute `(Hp,Wp)=(H/patch,W/patch)`. If you will pixel-shuffle by `s`, return `(Gh,Gw)=(Hp/s, Wp/s)`. Add `Normalize(mean,std)`.

**Signature change**: pass `vit_patch_size` and `pixel_shuffle_factor`.

**Patch**:

```diff
@@
-from data.custom_transforms import DynamicResize, SplitImage
+from data.custom_transforms import DynamicResize, SplitImage
+from torchvision.transforms import Normalize
+IMAGENET_MEAN = (0.485, 0.456, 0.406)
+IMAGENET_STD  = (0.229, 0.224, 0.225)
@@
-def get_image_processor(max_img_size, splitted_image_size, single_image_mode=False):
+def get_image_processor(
+    max_img_size,
+    splitted_image_size,
+    single_image_mode=False,
+    vit_patch_size=16,
+    pixel_shuffle_factor=2,
+    allow_upscale=True,
+):
@@
-    if single_image_mode:
-        # For DINOv3: resize entire image to 224x224, which will be processed into 14x14 patches
-        return transforms.Compose(
-            [
-                transforms.Resize((splitted_image_size, splitted_image_size)),
-                transforms.ToTensor(),
-                lambda x: (x, (1, 1)),  # Return tensor and grid count (1x1)
-            ]
-        )
+    if single_image_mode:
+        # DINOv3 path: aspect-preserving, snap both sides to multiples of patch, optional cap by max_img_size.
+        def _proc(pil):
+            t = transforms.Compose([
+                DynamicResize(patch_size=vit_patch_size, max_side_len=max_img_size, allow_upscale=allow_upscale),
+                transforms.ToTensor(),
+                Normalize(IMAGENET_MEAN, IMAGENET_STD),
+            ])
+            x = t(pil)                         # C,H,W
+            _, H, W = x.shape
+            Hp, Wp = H // vit_patch_size, W // vit_patch_size
+            s = pixel_shuffle_factor
+            assert Hp % s == 0 and Wp % s == 0, "Hp,Wp must be divisible by pixel_shuffle_factor"
+            Gh, Gw = Hp // s, Wp // s
+            return x, (Gh, Gw)
+        return _proc
@@
-    return transforms.Compose(
+    return transforms.Compose(
         [
             DynamicResize(splitted_image_size, max_img_size),
             transforms.ToTensor(),
+            Normalize(IMAGENET_MEAN, IMAGENET_STD),
             SplitImage(splitted_image_size),
         ]
     )
```

**Call sites to update**:

* `train.py` lines \~132–137:

```diff
-    image_processor = get_image_processor(
-        vlm_cfg.max_img_size, vlm_cfg.vit_img_size, single_image_mode
-    )
+    image_processor = get_image_processor(
+        vlm_cfg.max_img_size,
+        vlm_cfg.vit_img_size,
+        single_image_mode=single_image_mode,
+        vit_patch_size=vlm_cfg.vit_patch_size,
+        pixel_shuffle_factor=vlm_cfg.mp_pixel_shuffle_factor,
+        allow_upscale=True,
+    )
```

* `inference_example.py` lines \~57–65:

```diff
-        image_processor = get_image_processor(
-            proc_config["max_img_size"],
-            proc_config["vit_img_size"],
-            proc_config["single_image_mode"],
-        )
+        image_processor = get_image_processor(
+            proc_config["max_img_size"],
+            proc_config["vit_img_size"],
+            single_image_mode=proc_config["single_image_mode"],
+            vit_patch_size=proc_config.get("vit_patch_size", 16),
+            pixel_shuffle_factor=proc_config.get("mp_pixel_shuffle_factor", 2),
+        )
@@
-        image_processor = get_image_processor(
-            vlm_cfg.max_img_size, vlm_cfg.vit_img_size, single_image_mode
-        )
+        image_processor = get_image_processor(
+            vlm_cfg.max_img_size,
+            vlm_cfg.vit_img_size,
+            single_image_mode=single_image_mode,
+            vit_patch_size=vlm_cfg.vit_patch_size,
+            pixel_shuffle_factor=vlm_cfg.mp_pixel_shuffle_factor,
+        )
```

* `eval/benchmark_inference.py` and `eval/benchmark_suite.py` constructor calls: same update.

---

# 4) Wire `(Gh,Gw)` through the model

You need to pass grids into MP.

**Where**:

* `models/vision_language_model.py` (not in the merged file, but referenced everywhere).
  Update the forward where you currently:

1. run `vision_encoder(images)` → `[B, Seq, D]`
2. run `self.MP(feats)`.

**Change**: compute `Gh,Gw` per sample and call `self.MP(feats, gh=Gh, gw=Gw)`. If your forward treats `image_processor` output uniformly, you already have `(Gh,Gw)` in the batch. If not, carry it in the sample dict.

Pseudo-diff:

```diff
- feats = self.vision_encoder(image_tensor)             # B, Seq, D
- vis = self.MP(feats)                                  # assumes square
+ feats = self.vision_encoder(image_tensor)             # B, Seq, D
+ vis_list = []
+ for b in range(feats.size(0)):
+     Gh, Gw = grids[b]
+     vis_list.append(self.MP(feats[b:b+1], gh=Gh, gw=Gw))
+ vis = torch.cat(vis_list, dim=0)                      # B, Gh*Gw, D_lm
```

---

# 5) Remove the fixed “49 tokens per image” assumption

**File**: `models/config.py`
**Lines**: \~221–225

Issue: `mp_image_token_length=49` assumes 224×224, p=16, s=2. This explodes or underflows token counts at other sizes.

**Fix**:

* For DINOv3 set `mp_image_token_length=1`.
* Make `get_image_string` emit exactly `Gh×Gw` placeholders. You already loop over grids; stop multiplying by `mp_image_token_length` in the DINOv3 path.

Two options:

**Option A (clean)**: delete the `* mp_image_token_length` multiplier for all cases.

```diff
@@ def get_image_string(...):
-                image_string += tokenizer.image_token * mp_image_token_length
+                image_string += tokenizer.image_token
```

**Option B (minimal)**: set `mp_image_token_length=1` when `vit_architecture=="dinov3"`.

I recommend A. It aligns text placeholders with the actual visual token count.

---

# 6) Fix scripts that assume tensors

**File**: `eval/benchmark-inference.py`
**Lines**: \~34–38

Issue: calling `.unsqueeze(0)` on a tuple.

**Patch**:

```diff
-    image = image_processor(image)
-    image = image.unsqueeze(0).to(device)
+    processed = image_processor(image)       # either tensor or (tensor, (Gh,Gw))
+    if isinstance(processed, tuple):
+        image, grid = processed
+    else:
+        image, grid = processed, (1,1)
+    image = image.unsqueeze(0).to(device)
```

Do the same in `eval/benchmark_suite.py` load path.

Your `inference_example.py` already handles tuples correctly.

---

# 7) Label padding

**File**: `data/collators.py`
**Lines**: BaseCollator `_pad_batch`, \~10–30

Issue: BaseCollator pads labels with `pad_token_id`. You fixed this in `VQACollator`, but BaseCollator remains foot-gun.

**Patch**:

```diff
@@ class BaseCollator(object):
-        batch["labels"] = [
-            torch.nn.functional.pad(
-                labels, (max_length - len(labels), 0), value=self.tokenizer.pad_token_id
-            )
-            for labels in batch["labels"]
-        ]
+        batch["labels"] = [
+            torch.nn.functional.pad(labels, (max_length - len(labels), 0), value=-100)
+            for labels in batch["labels"]
+        ]
```

---

# 8) Normalization

Handled in section 3 via `Normalize(IMAGENET_MEAN, IMAGENET_STD)` in both single-image and split-image paths. Without this your encoder underperforms.

---

# 9) RoPE precision

Already acceptable: in `DINOv3RoPEPositionEmbedding.forward` you compute cos/sin with `torch.float32` and cast to `dtype` at the end (lines \~128–135). Keep that. Ensure AMP is enabled in training where you want it.

---

# 10) CLS and registers

**File**: `models/modality_projector.py`
You correctly drop CLS and register tokens when `mp_handle_special_tokens=True` (lines \~63–75). Keep this. Optionally, append the CLS token back after pixel shuffle if you want one global visual token. If so, project it with a separate linear and concat. Not required.

---

# 11) Token budget control (recommended)

Add a simple cap so you do not overflow LM context with very large images.

**Config**: add `vlm_max_image_tokens: int = 1024`.

**Runtime policy**: choose a per-sample downsampling factor `s` so that `(Hp/s)*(Wp/s) ≤ vlm_max_image_tokens`. That requires passing `s` into both the processor (to compute `(Gh,Gw)`) and the MP at runtime. If you want this now:

* Let `get_image_processor` accept `pixel_shuffle_factor` **or** a function `choose_factor(Hp,Wp)`.
* Compute `s` inside the per-sample call.
* Store `s` with the grid so MP uses the same factor.

I would make this a second pass after items 1–6.

---

# 12) Dataloader bucketing

Batch images with similar H×W to reduce padding and improve throughput. You already have knapsack controls in config. Add image-size buckets (e.g., by `longest_side ∈ {512, 768, 1024}`) at dataset collation if you see OOMs.

---

# 13) Minor correctness items

* **`get_image_string` grid-token check**
  **File**: `data/processors.py` lines \~100–107
  `hasattr(tokenizer, grid_token_name)` is not a real test for added tokens. Use

  ```py
  if grid_token_name in tokenizer.get_vocab():
      image_string += grid_token_name
  else:
      image_string += tokenizer.image_token
  ```

  or keep the generic token only.

* **`models/vision_transformer.from_pretrained` mapping**
  The mapping block is fine conceptually. After enabling RoPE, verify your HF DINOv3 checkpoints still load and forward works on arbitrary H×W.

---

# Sanity tests to add

1. **Rectangular input**: B=1, image 640×960. Expect `Hp=40,Wp=60`. With `s=2`, MP output tokens = 20×30.
2. **Grid identity**: Run MP then an inverse reshape test to ensure token count and ordering are stable.
3. **RoPE functional**: Shift the image by 16px horizontally, confirm attention logits change meaningfully.
4. **Throughput**: 224 vs 768 long-side. Confirm linear rise in tokens and memory.
5. **Normalization**: disable Normalize and observe clear drop in linear probe accuracy to catch accidental regressions.

---

# Summary of concrete edits

* `models/vision_transformer.py`: enable RoPE for DINOv3; set `vit_use_rope=True`; remove the `sqrt` grid inference; define `use_dinov3_rope`; generate RoPE from true `(Hp,Wp)`.
* `models/modality_projector.py`: add `pixel_shuffle_2d` and `forward(..., gh, gw)`.
* `data/processors.py`: dynamic DINOv3 path with `DynamicResize` + `Normalize`; return `(tensor, (Gh,Gw))`; update signature to carry `vit_patch_size` and `pixel_shuffle_factor`.
* `models/config.py`: set `mp_image_token_length=1` or remove multiplier usage; keep `mp_handle_special_tokens=True`.
* `eval/benchmark-inference.py` and `eval/benchmark_suite.py`: unpack tuples from the processor.
* `data/collators.py`: use `-100` in `BaseCollator` too.

This set gets you correct, high-res DINOv3 behavior, stable token accounting, and better performance on OCR-like tasks.



