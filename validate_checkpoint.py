#!/usr/bin/env python
"""
Run validation on a saved checkpoint using the training validation dataset.
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def generate_sample_outputs(model, dataset, tokenizer, checkpoint_path, device):
    """Generate text outputs from 5 randomly selected samples."""

    # Randomly select 5 samples from the dataset
    sample_indices = random.sample(range(len(dataset)), min(5, len(dataset)))

    sample_outputs = []

    for idx in sample_indices:
        try:
            sample = dataset[idx]

            # Get input tensors
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            images = sample["images"]

            # Handle images properly - convert list to tensor if needed
            if isinstance(images, list):
                if images:
                    # Stack images for batch processing
                    images = torch.stack(images).to(device)
                    # Ensure we have [batch_size, num_images, C, H, W] or [batch_size, C, H, W]
                    if images.dim() == 3:  # [C, H, W] - single image
                        images = images.unsqueeze(0)  # [1, C, H, W]
                    elif images.dim() == 4:  # [num_images, C, H, W]
                        if images.shape[0] == 1:
                            # Single image case [1, C, H, W]
                            images = images.squeeze(0).unsqueeze(0)  # [1, C, H, W]
                        else:
                            # Multi-image case - model expects [batch_size, C, H, W] for single sample
                            # For now, just use the first image to avoid the conv2d error
                            images = images[0:1]  # Take first image [1, C, H, W]
                else:
                    # No images case
                    images = torch.empty(
                        0,
                        model.cfg.vit_channels,
                        model.cfg.vit_img_size,
                        model.cfg.vit_img_size,
                        device=device,
                    )
            elif isinstance(images, torch.Tensor):
                # Single image - add batch dimension if needed
                if images.dim() == 3:
                    images = images.unsqueeze(0).to(device)
                elif images.dim() == 4:
                    # If it's [1, C, H, W], keep as is
                    images = images.to(device)
                elif images.dim() == 5:
                    # If it's [1, 1, C, H, W], squeeze out the extra dim
                    images = images.squeeze(1).to(device)

            # Decode the prompt
            prompt_tokens = sample["input_ids"].tolist()
            prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)

            # Generate text
            try:
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids,
                        images,
                        max_new_tokens=30,
                        temperature=0.7,
                        top_p=0.9,
                    )

                # Decode generated text
                generated_text = tokenizer.decode(
                    generated_ids[0].tolist(), skip_special_tokens=True
                )

                # Remove the prompt from generated text if present
                if generated_text.startswith(prompt_text):
                    generated_text = generated_text[len(prompt_text) :].strip()

            except Exception as gen_error:
                print(f"  ! Generation error for sample {idx}: {gen_error}")
                generated_text = f"[Generation failed: {str(gen_error)[:100]}]"

            sample_outputs.append(
                {
                    "sample_index": idx,
                    "prompt": prompt_text[:200] + "..."
                    if len(prompt_text) > 200
                    else prompt_text,
                    "generated_text": generated_text[:200] + "..."
                    if len(generated_text) > 200
                    else generated_text,
                }
            )

            print(f"  ✓ Generated output {len(sample_outputs)}/5")

        except Exception as e:
            print(f"  ✗ Error processing sample {idx}: {e}")
            continue

    # Save sample outputs
    output_path = Path(checkpoint_path) / "sample_outputs.json"
    with open(output_path, "w") as f:
        json.dump(sample_outputs, f, indent=2, ensure_ascii=False)

    print(f"✓ Generated {len(sample_outputs)} sample outputs saved to {output_path}")


def get_parser():
    """Create and return the argument parser."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run validation on a saved checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the checkpoint directory (e.g., checkpoints/model_name/step_3000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for validation",
    )
    parser.add_argument(
        "--limit_batches",
        type=int,
        default=None,
        help="Limit validation to N batches for quick testing (default: run all)",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint_path)

    print(f"Loading model from {checkpoint_path}...")

    from data.collators import VQACollator
    from data.datasets import VQADataset
    from data.processors import get_image_processor, get_tokenizer
    from models.config import TrainConfig
    from models.vision_language_model import VisionLanguageModel

    # Load model
    model = VisionLanguageModel.from_pretrained(str(checkpoint_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(
        f"✓ Model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Get config from model
    cfg = model.cfg
    train_cfg = TrainConfig()

    # Load tokenizer and processor
    tokenizer_path = checkpoint_path / "tokenizer"
    processor_path = checkpoint_path / "processor"

    if tokenizer_path.exists():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        print("✓ Tokenizer loaded from checkpoint")
    else:
        tokenizer = get_tokenizer(
            cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template
        )
        print(f"✓ Tokenizer loaded from HuggingFace: {cfg.lm_tokenizer}")

    if processor_path.exists():
        from transformers import AutoProcessor

        image_processor = AutoProcessor.from_pretrained(str(processor_path))
        print("✓ Image processor loaded from checkpoint")
    else:
        # Use single image mode for DINOv3 (resize to 224x224 instead of splitting)
        single_image_mode = cfg.vit_architecture == "dinov3"
        image_processor = get_image_processor(
            cfg.max_img_size, cfg.vit_img_size, single_image_mode
        )
        print(f"✓ Image processor loaded from HuggingFace: {cfg.vit_model_type}")

    # Load the SAME dataset as train.py uses
    print("\nLoading the_cauldron dataset (same as training)...")
    from datasets import concatenate_datasets, get_dataset_config_names, load_dataset

    # Load ALL dataset configs just like train.py
    combined_train_data = []
    dataset_names_to_load = train_cfg.train_dataset_name

    if "all" in dataset_names_to_load:
        dataset_names_to_load = get_dataset_config_names(train_cfg.train_dataset_path)

    for dataset_name in dataset_names_to_load:
        try:
            train_ds = load_dataset(
                train_cfg.train_dataset_path,
                dataset_name,
                num_proc=1,
            )
            train_ds["train"][0]  # Check if loaded correctly
            combined_train_data.append(train_ds["train"])
        except Exception as e:
            print(f"Warning: Failed to load '{dataset_name}': {e}")
            continue

    if not combined_train_data:
        raise ValueError("No valid datasets were loaded!")

    # Concatenate and shuffle with SAME seed as training
    train_ds = concatenate_datasets(combined_train_data)
    train_ds = train_ds.shuffle(seed=0)

    # Calculate split EXACTLY as train.py does
    total_samples = len(train_ds)
    val_size = int(total_samples * train_cfg.val_ratio)
    train_size = total_samples - val_size

    print(f"Dataset: {train_size} train, {val_size} validation samples")

    # Create validation dataset from the SAME split as training
    val_dataset = VQADataset(
        train_ds.select(range(train_size, total_samples)),
        tokenizer,
        image_processor,
        cfg.mp_image_token_length,
    )
    print(f"✓ Validation dataset loaded with {len(val_dataset)} samples")

    # Create dataloader
    # Use validation collator that truncates instead of discarding long sequences
    collator = VQACollator(tokenizer, max_length=cfg.lm_max_length, is_validation=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    # Run validation
    total_batches = len(val_loader)
    batches_to_process = args.limit_batches if args.limit_batches else total_batches

    print(f"\nRunning validation on {batches_to_process} batches...")
    print(
        f"This will process {len(val_dataset)} samples (batch_size={args.batch_size})"
    )

    import time

    start_time = time.time()
    total_val_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(val_loader, desc="Validation", total=batches_to_process)
        ):
            if args.limit_batches and batch_idx >= args.limit_batches:
                break
            
            # Skip None batches (shouldn't happen with truncation, but handle gracefully)
            if batch is None:
                continue

            input_ids = batch["input_ids"].to(device)
            images = batch["images"]
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            try:
                logits, loss = model(
                    input_ids,
                    images,
                    attention_mask=attention_mask,
                    targets=labels,
                )

                if loss is not None:
                    total_val_loss += loss.item()
                    num_batches += 1
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                continue

    elapsed_time = time.time() - start_time

    if num_batches > 0:
        avg_val_loss = total_val_loss / num_batches
        print(f"\n✓ Validation completed in {elapsed_time:.1f} seconds")
        print(f"  Average validation loss: {avg_val_loss:.4f}")
        print(f"  Batches processed: {num_batches}/{total_batches}")
        print(
            f"  Samples processed: {num_batches * args.batch_size}/{len(val_dataset)}"
        )

        # Save results
        output_path = checkpoint_path / "validation_results.json"
        results = {
            "checkpoint_path": str(checkpoint_path),
            "num_batches": num_batches,
            "num_samples": num_batches * args.batch_size,
            "average_loss": avg_val_loss,
            "total_loss": total_val_loss,
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Validation results saved to {output_path}")

        # Generate and save sample outputs
        print("\nGenerating sample text outputs...")
        generate_sample_outputs(
            model, val_dataset, tokenizer, str(checkpoint_path), device
        )

        # Print summary
        print("\nValidation Summary:")
        print(f"  Checkpoint: {checkpoint_path.name}")
        print(f"  Batches processed: {num_batches}")
        print(f"  Average loss: {avg_val_loss:.4f}")
        print("  Results saved to: validation_results.json")
        print("  Sample outputs saved to: sample_outputs.json")
    else:
        print("\n✗ No valid batches processed")


if __name__ == "__main__":
    main()
