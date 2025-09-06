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
    else:
        # Fall back to loading from HF
        from data.processors import get_tokenizer
        tokenizer = get_tokenizer(
            model.cfg.lm_tokenizer, 
            model.cfg.vlm_extra_tokens, 
            model.cfg.lm_chat_template
        )
    
    # Load/create image processor
    processor_path = checkpoint_path / "processor"
    if (processor_path / "preprocessor_config.json").exists():
        # Load our custom config
        with open(processor_path / "preprocessor_config.json", "r") as f:
            proc_config = json.load(f)
        
        image_processor = get_image_processor(
            proc_config["max_img_size"],
            proc_config["vit_img_size"],
            proc_config["single_image_mode"]
        )
    else:
        # Create from model config
        single_image_mode = model.cfg.vit_architecture == "dinov3"
        image_processor = get_image_processor(
            model.cfg.max_img_size,
            model.cfg.vit_img_size,
            single_image_mode
        )
    
    return model, tokenizer, image_processor, device


def generate_response(model, tokenizer, image_processor, image_path, prompt, device):
    """Generate a response for an image-text prompt."""
    
    # Load and process image
    image = Image.open(image_path).convert("RGB")
    processed = image_processor(image)
    
    if isinstance(processed, tuple):
        images, _ = processed
    else:
        images = processed
    
    # Handle tensor dimensions
    if isinstance(images, list):
        images = torch.stack(images)
    if images.dim() == 3:
        images = images.unsqueeze(0)  # Add batch dimension
    images = images.to(device)
    
    # Tokenize prompt
    # Format as chat if using instruct model
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        input_text = prompt
    
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    
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
    if generated_text.startswith(input_text):
        generated_text = generated_text[len(input_text):].strip()
    
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
        model, tokenizer, image_processor, 
        args.image_path, args.prompt, device
    )
    
    print("\n" + "="*50)
    print("Model Response:")
    print("="*50)
    print(response)
    print("="*50)


if __name__ == "__main__":
    main()