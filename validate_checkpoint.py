#!/usr/bin/env python
"""
Run validation on a saved checkpoint using the training validation dataset.
"""

import json
import os
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    checkpoint_path = "checkpoints/nanoVLM_dinov3-vitb16-pretrain-lvd1689m_1024_mp2_gemma-3-270m-it_1xGPU_423791samples_bs64_5000_lr5e-05-0.00512_0905-221819/step_3000"
    
    print(f"Loading model from {checkpoint_path}...")
    
    from models.vision_language_model import VisionLanguageModel
    from models.config import VLMConfig, TrainConfig
    from data.collators import VQACollator
    from data.datasets import VQADataset
    from data.processors import get_tokenizer, get_image_processor
    
    # Load model
    model = VisionLanguageModel.from_pretrained(checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"✓ Model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Get config from model
    cfg = model.cfg
    train_cfg = TrainConfig()
    
    # Load tokenizer and processor
    tokenizer_path = os.path.join(checkpoint_path, "tokenizer")
    processor_path = os.path.join(checkpoint_path, "processor")
    
    if os.path.exists(tokenizer_path):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"✓ Tokenizer loaded from checkpoint")
    else:
        tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)
        print(f"✓ Tokenizer loaded from HuggingFace: {cfg.lm_tokenizer}")
    
    if os.path.exists(processor_path):
        from transformers import AutoProcessor
        image_processor = AutoProcessor.from_pretrained(processor_path)
        print(f"✓ Image processor loaded from checkpoint")
    else:
        # Use single image mode for DINOv3 (resize to 224x224 instead of splitting)
        single_image_mode = cfg.vit_architecture == "dinov3"
        image_processor = get_image_processor(cfg.max_img_size, cfg.vit_img_size, single_image_mode)
        print(f"✓ Image processor loaded from HuggingFace: {cfg.vit_model_type}")
    
    # Load the HuggingFace dataset
    print("\nLoading HuggingFace dataset...")
    from datasets import load_dataset
    
    # Load a small subset for validation - use vqav2 as it's a common VQA dataset
    train_ds = load_dataset(
        train_cfg.train_dataset_path,
        "vqav2",  # Use vqav2 subset for testing
        split="train",
        streaming=True,
    )
    
    # Convert to list and take a small subset for validation
    samples = []
    for i, sample in enumerate(train_ds):
        if i >= 100:  # Take only 100 samples for quick validation
            break
        samples.append(sample)
    
    from datasets import Dataset
    val_ds = Dataset.from_list(samples)
    
    # Create validation dataset
    print("Creating validation dataset...")
    val_dataset = VQADataset(
        val_ds,
        tokenizer,
        image_processor,
        cfg.mp_image_token_length,
    )
    print(f"✓ Validation dataset loaded with {len(val_dataset)} samples")
    
    # Create dataloader
    collator = VQACollator(tokenizer, max_length=cfg.lm_max_length)
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    
    # Run validation
    print("\nRunning validation...")
    total_val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
            if batch_idx >= 10:  # Limit to 10 batches for testing
                break
                
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
    
    if num_batches > 0:
        avg_val_loss = total_val_loss / num_batches
        print(f"\n✓ Validation completed")
        print(f"  Average validation loss: {avg_val_loss:.4f}")
        
        # Save results
        output_path = os.path.join(checkpoint_path, "validation_results.json")
        results = {
            "checkpoint_path": checkpoint_path,
            "num_batches": num_batches,
            "num_samples": num_batches * 4,  # batch_size=4
            "average_loss": avg_val_loss,
            "total_loss": total_val_loss,
        }
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Validation results saved to {output_path}")
        
        # Print summary
        print("\nValidation Summary:")
        print(f"  Checkpoint: {Path(checkpoint_path).name}")
        print(f"  Batches processed: {num_batches}")
        print(f"  Average loss: {avg_val_loss:.4f}")
        print(f"  Results saved to: validation_results.json")
    else:
        print("\n✗ No valid batches processed")

if __name__ == "__main__":
    main()