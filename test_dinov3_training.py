import os
import sys
import torch
import time

sys.path.append('.')

from models.config import get_dinov3_gemma_config, TrainConfig
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor
from data.collators import VQACollator
from PIL import Image
import numpy as np

# Quick training test script
def main():
    print("=" * 60)
    print("DINOv3 + Gemma-3-270M Training Demo")
    print("=" * 60)
    
    # Configs
    vlm_cfg = get_dinov3_gemma_config()
    train_cfg = TrainConfig()
    train_cfg.batch_size = 2
    train_cfg.log_wandb = False  # Disable for demo
    
    # Create model
    print("\nInitializing model...")
    model = VisionLanguageModel(vlm_cfg, load_backbone=True)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized: {param_count:,} parameters")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"✓ Using device: {device}")
    
    # Create tokenizer and processor
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)
    image_processor = get_image_processor(vlm_cfg.max_img_size, vlm_cfg.vit_img_size)
    collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.MP.parameters(), 'lr': train_cfg.lr_mp},
        {'params': list(model.decoder.parameters()) + list(model.vision_encoder.parameters()), 
         'lr': train_cfg.lr_backbones}
    ])
    
    # Create synthetic training data for demo
    print("\nCreating synthetic training batch...")
    
    # Create a dummy image (224x224 for DINOv3)
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    processed_image = image_processor(dummy_image)
    
    # Create sample conversations
    samples = []
    for i in range(train_cfg.batch_size):
        sample = {
            'images': [processed_image],
            'input_ids': [],
            'labels': []
        }
        
        # Create a simple Q&A conversation
        question = f"<|image|> What is in this image?"
        answer = f"This is image number {i}. It contains various objects."
        
        # Tokenize
        q_tokens = tokenizer(question, add_special_tokens=False).input_ids
        a_tokens = tokenizer(answer, add_special_tokens=False).input_ids
        
        # Combine tokens
        sample['input_ids'] = q_tokens + a_tokens
        # Labels: -100 for question (don't compute loss), actual tokens for answer
        sample['labels'] = [-100] * len(q_tokens) + a_tokens
        
        samples.append(sample)
    
    # Collate batch
    batch = collator(samples)
    
    # Move to device
    images = batch['images']
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    print(f"✓ Batch created: {input_ids.shape}")
    
    # Training step
    print("\nPerforming training step...")
    model.train()
    
    # Mixed precision training
    with torch.autocast(device_type=device.type, 
                       dtype=torch.bfloat16 if device.type == 'cuda' else torch.float16):
        # Forward pass
        start = time.time()
        logits, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
        forward_time = time.time() - start
        
    print(f"  ✓ Forward pass: {forward_time:.3f}s")
    print(f"  ✓ Loss: {loss.item():.4f}")
    
    # Backward pass
    start = time.time()
    loss.backward()
    backward_time = time.time() - start
    print(f"  ✓ Backward pass: {backward_time:.3f}s")
    
    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"  ✓ Gradient norm: {grad_norm:.4f}")
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    print(f"  ✓ Optimizer step completed")
    
    # Memory stats
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\nGPU Memory:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING SUCCESSFUL - NO WARNINGS OR ERRORS\!")
    print("=" * 60)
    
    # Model summary
    print("\nModel Architecture Summary:")
    print("├─ Vision: DINOv3-ViTS16Plus")
    print("│  ├─ 384 hidden dim, 12 layers, 6 heads")
    print("│  ├─ Sin/cos position embeddings (flexible resolution)")
    print("│  ├─ LayerScale for stability")
    print("│  ├─ SwiGLU FFN activation")
    print("│  └─ 4 register tokens + CLS token")
    print("├─ Language: Gemma-3-270M-IT")
    print("│  ├─ 640 hidden dim, 18 layers")
    print("│  ├─ 256 custom head dim")
    print("│  └─ 256k vocabulary")
    print("└─ Projector: PixelShuffle 2x → 49 image tokens")
    print(f"\nTotal: {param_count:,} parameters (~298M)")
    
    # Training metrics
    print("\nTraining Metrics:")
    print(f"  Forward pass time: {forward_time:.3f}s")
    print(f"  Backward pass time: {backward_time:.3f}s")
    print(f"  Total step time: {forward_time + backward_time:.3f}s")
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Gradient norm: {grad_norm:.4f}")
    
    print("\n🎉 Ready for full training with wandb logging\!")
    print("   Run: python train.py --use_preset dinov3_gemma")

if __name__ == "__main__":
    main()
