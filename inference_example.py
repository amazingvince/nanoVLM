#!/usr/bin/env python
"""
Example script showing how to use a trained nanoVLM model for inference.
This demonstrates the intended usage pattern for users after training.
"""

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer

from data.processors import get_image_processor
from models.vision_language_model import VisionLanguageModel


def load_nanovlm_model(checkpoint_path):
    """Load a trained nanoVLM model with its tokenizer and processor."""
    checkpoint_path = Path(checkpoint_path)

    # Load the model
    model = VisionLanguageModel.from_pretrained(str(checkpoint_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer_path = checkpoint_path / "tokenizer"
    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        # Add image_token_id attribute if missing (needed for model)
        if not hasattr(tokenizer, "image_token_id"):
            image_token = model.cfg.vlm_extra_tokens.get("image_token", "<|image|>")
            tokenizer.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
            tokenizer.image_token = image_token
    else:
        # Fall back to loading from HF
        from data.processors import get_tokenizer

        tokenizer = get_tokenizer(
            model.cfg.lm_tokenizer,
            model.cfg.vlm_extra_tokens,
            model.cfg.lm_chat_template,
        )

    # Set tokenizer in model (needed for image token replacement)
    model.tokenizer = tokenizer

    # Load/create image processor
    processor_path = checkpoint_path / "processor"
    if (processor_path / "preprocessor_config.json").exists():
        # Load our custom config
        with open(processor_path / "preprocessor_config.json", "r") as f:
            proc_config = json.load(f)

        image_processor = get_image_processor(
            proc_config["max_img_size"],
            proc_config["vit_img_size"],
            proc_config["single_image_mode"],
        )
    else:
        # Create from model config
        single_image_mode = model.cfg.vit_architecture == "dinov3"
        image_processor = get_image_processor(
            model.cfg.max_img_size, model.cfg.vit_img_size, single_image_mode
        )

    return model, tokenizer, image_processor, device


def generate_response(model, tokenizer, image_processor, image_path, prompt, device):
    """Generate a response for an image-text prompt."""

    # Load and process image
    image = Image.open(image_path).convert("RGB")
    processed = image_processor(image)

    if isinstance(processed, tuple):
        images, grid_size = processed
    else:
        images = processed
        grid_size = (1, 1)  # Single image for DINOv3

    # Handle tensor dimensions
    if isinstance(images, list):
        images = torch.stack(images)
    if images.dim() == 3:
        images = images.unsqueeze(0)  # Add batch dimension
    images = images.to(device)

    # Calculate number of image tokens
    vit_patch_size = grid_size[0] * grid_size[1]

    # Get image token from tokenizer or model config
    if hasattr(tokenizer, "image_token"):
        image_token = tokenizer.image_token
    else:
        # Use the image token from model config
        image_token = model.cfg.vlm_extra_tokens.get("image_token", "<|image|>")

    image_token_str = image_token * model.cfg.mp_image_token_length * vit_patch_size

    # Format prompt with image tokens
    full_prompt = image_token_str + prompt

    # Format as chat template
    messages = [{"role": "user", "content": full_prompt}]

    # Use chat template or fallback to direct format
    input_text = None
    try:
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
    except Exception:
        # Fallback if chat template not available
        input_text = (
            f"<|im_start|>user\n{full_prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    # Ensure input_ids is 2D
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            images,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
        )

    # Decode response
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Remove the prompt from generated text if present
    if input_text and generated_text.startswith(input_text):
        generated_text = generated_text[len(input_text) :].strip()

    return generated_text


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run inference with a trained nanoVLM model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to checkpoint directory (e.g., checkpoints/model/step_1000)",
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to input image",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="Text prompt for the model",
    )

    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint_path}...")
    model, tokenizer, image_processor, device = load_nanovlm_model(args.checkpoint_path)
    print(f"✓ Model loaded on {device}")

    print(f"\nProcessing image: {args.image_path}")
    print(f"Prompt: {args.prompt}")
    print("\nGenerating response...")

    response = generate_response(
        model, tokenizer, image_processor, args.image_path, args.prompt, device
    )

    print("\n" + "=" * 50)
    print("Model Response:")
    print("=" * 50)
    print(response)
    print("=" * 50)


if __name__ == "__main__":
    main()
