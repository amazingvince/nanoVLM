import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

from data.processors import get_tokenizer
from models.config import VLMConfig
from models.language_model import LanguageModel
from models.modality_projector import ModalityProjector
from models.utils import top_k_top_p_filtering
from models.vision_encoder_registry import create_vision_encoder
from models.vision_encoder_base import VisionEncoderOutput

# Import encoders to register them
import models.encoders  # noqa: F401


class VisionLanguageModel(nn.Module):
    """Vision-Language Model combining vision encoder, language decoder, and modality projector.
    
    :param cfg: VLMConfig containing model configuration
    :param load_backbone: Whether to load pretrained backbone weights
    """
    def __init__(self, cfg: VLMConfig, load_backbone: bool = True):
        super().__init__()
        self.cfg = cfg

        # Create vision encoder using factory (updates cfg with encoder-specific values)
        encoder_type = getattr(cfg, "vision_encoder_type", "siglip")
        print(f"Using vision encoder: {encoder_type}")
        self.vision_encoder = create_vision_encoder(cfg, load_pretrained=load_backbone)

        if load_backbone:
            print("Loading from backbone weights")
            self.decoder = LanguageModel.from_pretrained(cfg)
        else:
            self.decoder = LanguageModel(cfg)

        self.MP = ModalityProjector(cfg)
        self.load_backbone = load_backbone
        self.tokenizer = get_tokenizer(
            cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template
        )

    def _replace_img_tokens_with_embd(
        self, input_ids: torch.Tensor, token_embd: torch.Tensor, image_embd: torch.Tensor
    ) -> torch.Tensor:
        """Replace image-token placeholders with actual image embeddings.
        
        :param input_ids: Token IDs [batch_size, seq_len]
        :param token_embd: Token embeddings [batch_size, seq_len, hidden_dim]
        :param image_embd: Image embeddings [num_images, num_patches, hidden_dim]
        :return: Updated token embeddings with image embeddings inserted
        """
        # Clone the original embeddings to avoid in-place issues
        updated_token_embd = token_embd.clone()

        # Build a mask of all image-token positions: shape [B, T_seq]
        mask = input_ids == self.tokenizer.image_token_id
        updated_token_embd[mask] = image_embd.view(-1, image_embd.size(-1)).to(
            updated_token_embd.dtype
        )  # torch flattens before assigning

        return updated_token_embd

    def _process_images(
        self, images: Union[torch.Tensor, List[torch.Tensor]], device: torch.device
    ) -> Optional[torch.Tensor]:
        """Process and concatenate images into a single tensor.
        
        :param images: Input images as tensor or list of tensors
        :param device: Target device for tensor
        :return: Concatenated image tensor or None if no images
        """
        if isinstance(images, list):
            if images and isinstance(images[0], list):
                images = [img for sublist in images for img in sublist]

            if not images:  # Handle cases with no images
                return None
            else:
                return torch.cat(images, dim=0).to(device)
        return images  # Already a tensor

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Union[torch.Tensor, List[torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the vision-language model.
        
        :param input_ids: Input token IDs [batch_size, seq_len]
        :param images: Images as tensor or list [batch_size, 3, H, W]
        :param attention_mask: Attention mask [batch_size, seq_len]
        :param targets: Target token IDs for computing loss [batch_size, seq_len]
        :return: Tuple of (logits [batch_size, seq_len, vocab_size], loss)
        """
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids)  # [B, T_sequence, D_lm]

        if images_tensor is not None:
            # Get encoder output (now returns VisionEncoderOutput)
            encoder_output = self.vision_encoder(images_tensor)
            # Extract patch features (excluding CLS/register tokens if present)
            image_embd = encoder_output.features
            image_embd = self.MP(
                image_embd
            )  # [num_images, mp_image_token_length, D_lm]
            token_embd = self._replace_img_tokens_with_embd(
                input_ids, token_embd, image_embd
            )

        logits, _ = self.decoder(token_embd, attention_mask=attention_mask)

        loss = None
        if targets is not None:
            logits = self.decoder.head(logits)  # Apply LM head
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
        input_ids: torch.Tensor,
        images: Union[torch.Tensor, List[torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 5,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 0.5,
        greedy: bool = False,
    ) -> torch.Tensor:
        """Generate text autoregressively given image and text inputs.
        
        :param input_ids: Input token IDs [batch_size, seq_len]
        :param images: Images as tensor or list
        :param attention_mask: Attention mask
        :param max_new_tokens: Number of tokens to generate
        :param top_k: Top-k sampling parameter
        :param top_p: Top-p (nucleus) sampling parameter
        :param temperature: Temperature for sampling
        :param greedy: Whether to use greedy decoding
        :return: Generated token IDs [batch_size, num_generated]
        """
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids)  # [B, T_prompt_text, D_lm]

        if images_tensor is not None:
            # 1. Process image if present
            # Get encoder output (now returns VisionEncoderOutput)
            encoder_output = self.vision_encoder(images_tensor)
            # Extract patch features (excluding CLS/register tokens if present)
            image_embd = encoder_output.features  # [B, T_img_feat, D_model]
            image_embd = self.MP(image_embd)  # [B, mp_image_token_length, D_lm]
            # 2. Combine image and text embeddings
            token_embd = self._replace_img_tokens_with_embd(
                input_ids, token_embd, image_embd
            )

        current_total_seq_len = token_embd.size(1)
        batch_size = input_ids.size(0)  # Or token_embd.size(0)

        # --- Multimodal Prefill Phase ---
        prefill_output, kv_cache_list = self.decoder(
            token_embd,
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
        local_path = Path(repo_id_or_path)
        if local_path.exists():
            config_path = local_path / "config.json"
            weights_path = local_path / "model.safetensors"

            if not config_path.exists():
                raise ValueError(
                    f"Config file not found at {config_path}. Please provide a valid path."
                )
            if not weights_path.exists():
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
            cfg = VLMConfig(**json.load(f))

        # Initialize model without loading the backbone
        model = cls(cfg, load_backbone=False)

        # Load safetensors weights
        load_model(model, str(weights_path))

        # Done!
        return model

    def save_pretrained(self, save_directory: str) -> None:
        """Save model weights and configuration to directory.
        
        :param save_directory: Directory path to save model
        """
        # Create directory if it doesn't exist
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(save_path / "config.json", "w") as f:
            f.write(json.dumps(asdict(self.cfg), indent=4))

        # Save weights as safetensors
        save_model(self, str(save_path / "model.safetensors"))

    def push_to_hub(self, repo_id: str, private: bool = False) -> str:
        """Push model to Hugging Face Hub.
        
        :param repo_id: Repository ID on HuggingFace Hub
        :param private: Whether to create private repository
        :return: URL of the created/updated repository
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
            readme_path = Path(save_path) / "README.md"
            with open(readme_path, "w") as f:
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
