import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_model

from data.processors import get_tokenizer
from models.config import VLMConfig
from models.language_model import LanguageModel
from models.modality_projector import ModalityProjector
from models.utils import top_k_top_p_filtering
from models.vision_transformer import ViT


class VisionLanguageModel(nn.Module):
    def __init__(self, cfg: VLMConfig, load_backbone=True):
        super().__init__()
        self.cfg = cfg
        if load_backbone:
            print("Loading from backbone weights")
            self.vision_encoder = ViT.from_pretrained(cfg)
            self.decoder = LanguageModel.from_pretrained(cfg)
        else:
            self.vision_encoder = ViT(cfg)
            self.decoder = LanguageModel(cfg)
        self.MP = ModalityProjector(cfg)
        self.load_backbone = load_backbone
        self.tokenizer = get_tokenizer(
            cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template
        )

    def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
        """
        Replace every image-token placeholder in `input_ids` with the corresponding slice
        from `image_embd`. Supports an arbitrary number of image-token placeholders per sample.
        The first example in the batch might have 2 images and the second none.
        """
        # Clone the original embeddings to avoid in-place issues
        updated_token_embd = token_embd.clone()

        # Build a mask of all image-token positions: shape [B, T_seq]
        mask = input_ids == self.tokenizer.image_token_id

        # Count how many image tokens we have in the input
        num_image_tokens = mask.sum().item()

        # Flatten image embeddings
        flat_image_embd = image_embd.view(-1, image_embd.size(-1))

        # If we have more embeddings than tokens (due to truncation), only use what we need
        if flat_image_embd.size(0) > num_image_tokens:
            # Truncate image embeddings to match the number of image tokens
            flat_image_embd = flat_image_embd[:num_image_tokens]
        elif flat_image_embd.size(0) < num_image_tokens:
            # This shouldn't happen, but handle gracefully
            print(
                f"Warning: Expected {num_image_tokens} image embeddings but got {flat_image_embd.size(0)}"
            )
            # Pad with zeros if we somehow have fewer embeddings than tokens
            padding = torch.zeros(
                num_image_tokens - flat_image_embd.size(0),
                flat_image_embd.size(-1),
                dtype=flat_image_embd.dtype,
                device=flat_image_embd.device,
            )
            flat_image_embd = torch.cat([flat_image_embd, padding], dim=0)

        updated_token_embd[mask] = flat_image_embd.to(updated_token_embd.dtype)

        return updated_token_embd

    def forward(
        self, input_ids, images, attention_mask=None, targets=None, image_grids=None
    ):
        if isinstance(images, list):
            if not images:  # Handle cases with no images
                images = torch.empty(
                    0,
                    self.cfg.vit_channels,
                    self.cfg.vit_image_size,
                    self.cfg.vit_image_size,
                    device=input_ids.device,
                )
                image_grids = []
            else:
                if isinstance(images[0], list):
                    images = [img for sublist in images for img in sublist]

                # For DINOv3: Pad images to the same size before stacking
                # When we have variable-sized images due to aspect-preserving resize
                if images[0].dim() == 3:  # Individual images [C, H, W]
                    # Find max dimensions in the batch
                    max_h = max(img.shape[1] for img in images)
                    max_w = max(img.shape[2] for img in images)

                    # Store original sizes for proper grid handling
                    original_sizes = [(img.shape[1], img.shape[2]) for img in images]

                    # Pad each image to max dimensions (pad on right and bottom)
                    padded_images = []
                    for img in images:
                        pad_h = max_h - img.shape[1]
                        pad_w = max_w - img.shape[2]
                        # Pad format: (left, right, top, bottom) with zero padding
                        padded = F.pad(
                            img, (0, pad_w, 0, pad_h), mode="constant", value=0
                        )
                        padded_images.append(padded)

                    images = torch.stack(padded_images, dim=0).to(input_ids.device)

                    # Store original sizes for later use in modality projector
                    # This allows us to mask out padded tokens if needed
                    self._batch_original_sizes = original_sizes
                else:  # Already batched
                    images = torch.cat(images, dim=0).to(input_ids.device)

        # Process vision features
        image_embd = self.vision_encoder(images)
        # Apply modality projector with grid dimensions if available
        if image_grids is not None and len(image_grids) > 0:
            # Process each image separately with its grid
            projected = []

            # When images are padded to same size, we process them together
            # but need to handle grids individually based on original sizes
            if hasattr(self, "_batch_original_sizes"):
                # Images were padded - use original sizes to compute correct grids
                patch_size = self.cfg.vit_patch_size

                # Note: We don't need to track padded dimensions here
                # The modality projector will extract only the real tokens
                # based on the original (unpadded) dimensions we pass to it

                for i, (orig_h, orig_w) in enumerate(self._batch_original_sizes):
                    # Calculate actual patch grid for original image (before padding)
                    Hp = orig_h // patch_size
                    Wp = orig_w // patch_size

                    # Extract the full padded embeddings for this image
                    # The modality projector will handle extracting only real tokens
                    img_features = image_embd[i : i + 1]

                    # Pass original grid dimensions to modality projector
                    proj_embd = self.MP(img_features, gh=Hp, gw=Wp)
                    projected.append(proj_embd)

                # Clean up
                del self._batch_original_sizes
            else:
                # Original path for when images aren't padded
                for i, (gh, gw) in enumerate(image_grids):
                    # Get patch grid dimensions from vision encoder if stored
                    if hasattr(self.vision_encoder.patch_embedding, "_last_hw"):
                        Hp, Wp = self.vision_encoder.patch_embedding._last_hw
                    else:
                        # Fallback: compute from sequence length
                        seq_len = image_embd[i : i + 1].shape[1]
                        if self.cfg.vit_cls_flag:
                            seq_len -= 1
                        if hasattr(self.cfg, "vit_num_registers"):
                            seq_len -= self.cfg.vit_num_registers
                        Hp = Wp = int(seq_len**0.5)

                    # Pass grid dimensions directly from image_grids
                    proj_embd = self.MP(image_embd[i : i + 1], gh=gh, gw=gw)
                    projected.append(proj_embd)

            # Pad projected embeddings to same size before concatenation
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
        else:
            # Fallback to original square processing
            image_embd = self.MP(image_embd)

        token_embd = self.decoder.token_embedding(input_ids)  # [B, T_sequence, D_lm]

        updated_token_embd = self._replace_img_tokens_with_embd(
            input_ids, token_embd, image_embd
        )

        # The updated_token_embd is now the token_embd with image parts replaced.
        # The attention_mask comes from the collator and should already cover the full sequence.
        hidden_states, _ = self.decoder(
            updated_token_embd, attention_mask=attention_mask
        )

        # Always apply LM head to get logits
        logits = self.decoder.head(hidden_states)

        loss = None
        if targets is not None:
            # Loss is calculated over all tokens, but `targets` (labels) will have -100 for non-answer tokens.
            # No need to slice logits based on image embedding size here, as the target mask handles it.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        images,
        attention_mask=None,
        max_new_tokens=5,
        top_k=50,
        top_p=0.9,
        temperature=0.5,
        greedy=False,
        image_grids=None,
    ):
        if isinstance(images, list):
            if not images:  # Handle cases with no images
                images = torch.empty(
                    0,
                    self.cfg.vit_channels,
                    self.cfg.vit_image_size,
                    self.cfg.vit_image_size,
                    device=input_ids.device,
                )
                image_grids = []
            else:
                if isinstance(images[0], list):
                    images = [img for sublist in images for img in sublist]
                images = torch.cat(images, dim=0).to(input_ids.device)

        # 1. Process image
        image_embd = self.vision_encoder(images)  # [B, T_img_feat, D_model]

        print(f"DEBUG: image_grids type: {type(image_grids)}, len: {len(image_grids) if image_grids is not None else 'N/A'}")
        # Apply modality projector with grid dimensions if available
        if image_grids is not None and len(image_grids) > 0:
            # Process each image separately with its grid
            projected = []
            for i, (gh, gw) in enumerate(image_grids):
                # Get patch grid dimensions from vision encoder if stored
                if hasattr(self.vision_encoder.patch_embedding, "_last_hw"):
                    Hp, Wp = self.vision_encoder.patch_embedding._last_hw
                else:
                    # Fallback: compute from sequence length
                    seq_len = image_embd[i : i + 1].shape[1]
                    if self.cfg.vit_cls_flag:
                        seq_len -= 1
                    if hasattr(self.cfg, "vit_num_registers"):
                        seq_len -= self.cfg.vit_num_registers
                    Hp = Wp = int(seq_len**0.5)

                # Pass grid dimensions to modality projector
                proj_embd = self.MP(image_embd[i : i + 1], gh=Hp, gw=Wp)
                projected.append(proj_embd)

            # Pad projected embeddings to same size before concatenation
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
        else:
            # Fallback to original square processing
            image_embd = self.MP(image_embd)

        # 2. Embed initial text prompt tokens
        prompt_token_embeds = self.decoder.token_embedding(
            input_ids
        )  # [B, T_prompt_text, D_lm]

        # 3. Combine image and text embeddings
        initial_combined_embeds = self._replace_img_tokens_with_embd(
            input_ids, prompt_token_embeds, image_embd
        )

        current_total_seq_len = initial_combined_embeds.size(1)
        batch_size = input_ids.size(0)  # Or initial_combined_embeds.size(0)

        # --- Multimodal Prefill Phase ---
        prefill_output, kv_cache_list = self.decoder(
            initial_combined_embeds,
            attention_mask=attention_mask,  # Use the provided attention mask
            kv_cache=None,
            start_pos=0,
        )

        last_token_output_from_prefill = prefill_output[:, -1, :]

        if not self.decoder.lm_use_tokens:
            current_logits = self.decoder.head(last_token_output_from_prefill)
        else:
            current_logits = last_token_output_from_prefill

        # Store newly generated token IDs
        newly_generated_ids_list = []

        # --- Decode Phase by sampling tokens autoregressively using the kv-cache ---
        for _ in range(max_new_tokens):
            # Tell the compiler a new step is beginning
            torch.compiler.cudagraph_mark_step_begin()

            if greedy:
                next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(
                    current_logits, top_k=top_k, top_p=top_p
                )
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

            newly_generated_ids_list.append(next_token_id)

            # Embed the newly generated token
            next_token_embed = self.decoder.token_embedding(
                next_token_id
            )  # [B, 1, D_lm]

            # The start_pos for the new token is the current total sequence length *before* adding this new token
            current_token_start_pos = current_total_seq_len
            current_total_seq_len += 1

            # update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (batch_size, 1),
                            device=attention_mask.device,
                            dtype=attention_mask.dtype,
                        ),
                    ),
                    dim=1,
                )

            # With KV cache: only process the new token
            decode_step_output, kv_cache_list = self.decoder(
                next_token_embed,
                attention_mask=attention_mask,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos,
            )

            last_token_output = decode_step_output[:, -1, :]

            # Apply head to get logits (if model is in embedding mode)
            if not self.decoder.lm_use_tokens:
                current_logits = self.decoder.head(last_token_output)
            else:
                current_logits = last_token_output

        if not newly_generated_ids_list:  # Handle case where max_new_tokens might be 0
            return torch.empty(
                (batch_size, 0), dtype=torch.long, device=input_ids.device
            )

        generated_ids = torch.cat(newly_generated_ids_list, dim=1)

        # Post-process to handle EOS token.
        if (
            self.tokenizer.eos_token_id is not None and generated_ids.numel() > 0
        ):  # Ensure generated_ids is not empty
            seq_len = generated_ids.size(1)
            device = generated_ids.device

            eos_mask = (
                generated_ids == self.tokenizer.eos_token_id
            )  # Create a boolean mask for EOS tokens

            col_indices_for_min = torch.arange(
                seq_len, device=device
            )  # Create column indices [0, 1, ..., seq_len-1]

            # In eos_mask, mark positions with actual col_idx, others with a large number
            masked_col_indices = torch.where(
                eos_mask,
                col_indices_for_min.unsqueeze(0).expand_as(generated_ids),
                seq_len + 1,
            )

            first_eos_indices_values = torch.min(masked_col_indices, dim=1).values

            # Clamp values to seq_len (if no EOS found, min will be seq_len + 1, clamp brings it to seq_len0. This means if no EOS, or EOS is the last token, no replacement will happen for that sample.
            actual_first_eos_indices = torch.clamp(
                first_eos_indices_values, max=seq_len
            )

            # Create column indices for comparison, shape [batch_size, seq_len]
            col_indices_for_comparison = (
                torch.arange(seq_len, device=device)
                .unsqueeze(0)
                .expand_as(generated_ids)
            )

            # Tokens are replaced if their column index is greater than the index of the first EOS token
            replace_mask = (
                col_indices_for_comparison > actual_first_eos_indices.unsqueeze(1)
            )

            generated_ids[replace_mask] = self.tokenizer.eos_token_id

        return generated_ids

    @classmethod
    def from_pretrained(
        cls, repo_id_or_path: str, *, revision: Optional[str] = None
    ) -> "VisionLanguageModel":
        """
        Load a VisionLanguageModel from a local directory or a repo on the Hugging Face Hub.

        Args:
            repo_id_or_path (str): The path to the local directory or the Hugging Face Hub repo ID.

        Returns:
            VisionLanguageModel: The loaded model.
        """
        # If local folder exists => load from there
        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")

            if not os.path.exists(config_path):
                raise ValueError(
                    f"Config file not found at {config_path}. Please provide a valid path."
                )
            if not os.path.exists(weights_path):
                raise ValueError(
                    f"Weights file not found at {weights_path}. Please provide a valid path."
                )
        # Otherwise, assume it's a Hugging Face Hub repo
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="config.json", revision=revision
            )
            weights_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="model.safetensors", revision=revision
            )

        # Load config
        with open(config_path, "r") as f:
            config_dict = json.load(f)
            # Remove computed fields that shouldn't be in __init__
            config_dict.pop("lm_vocab_size", None)  # This is computed in __post_init__
            cfg = VLMConfig(**config_dict)

        # Initialize model without loading the backbone
        model = cls(cfg, load_backbone=False)

        # Load safetensors weights with strict=False to ignore buffer mismatches
        import safetensors.torch

        state_dict = safetensors.torch.load_file(weights_path)

        # Remove buffer entries that shouldn't be loaded (they're recomputed on init)
        keys_to_remove = []
        for key in state_dict.keys():
            if "rotary_embd.inv_freq" in key:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del state_dict[key]

        # Load the cleaned state dict
        model.load_state_dict(state_dict, strict=False)

        # Done!
        return model

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model and configuration to a directory.

        Args:
            save_directory (str): The directory to save the model and config.
        """
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Save config with all necessary fields
        config_dict = asdict(self.cfg)

        # Add computed fields that might be needed for loading
        # Check if lm_head_dim was set (for models like Gemma-3)
        if hasattr(self.cfg, "lm_head_dim"):
            config_dict["lm_head_dim"] = self.cfg.lm_head_dim

        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(json.dumps(config_dict, indent=4))

        # Save weights as safetensors
        save_model(self, os.path.join(save_directory, "model.safetensors"))

    def push_to_hub(self, repo_id: str, private: bool = False) -> None:
        """
        Push the model and configuration to the Hugging Face Hub.

        Args:
            repo_id (str): The repo ID on the Hugging Face Hub.
        """
        from huggingface_hub import create_repo, upload_folder

        # Create repo
        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print("Created repo: ", repo_url)

        with tempfile.TemporaryDirectory() as save_path:
            # Save to tmp directory
            self.save_pretrained(save_path)

            # Save model card
            with open(os.path.join(save_path, "README.md"), "w") as f:
                f.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

            # Upload
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload nanoVLM using push_to_hub",
            )


MODEL_CARD_TEMPLATE = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
library_name: nanovlm
license: mit
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - multimodal
  - research
---

**nanoVLM** is a minimal and lightweight Vision-Language Model (VLM) designed for efficient training and experimentation. Built using pure PyTorch, the entire model architecture and training logic fits within ~750 lines of code. It combines a ViT-based image encoder (SigLIP-B/16-224-85M) with a lightweight causal language model (SmolLM2-135M), resulting in a compact 222M parameter model.

For more information, check out the base model on https://huggingface.co/lusxvr/nanoVLM-222M.

**Usage:**

Clone the nanoVLM repository: https://github.com/huggingface/nanoVLM.
Follow the install instructions and run the following code:

```python
from models.vision_language_model import VisionLanguageModel

model = VisionLanguageModel.from_pretrained("{repo_id}")
```
"""
