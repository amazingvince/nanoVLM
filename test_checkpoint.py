#!/usr/bin/env python
"""
Direct evaluation of a checkpoint without using cli_evaluate.
"""

import json
import os
import sys
import torch
from pathlib import Path

# Test loading the checkpoint
checkpoint_path = "checkpoints/nanoVLM_dinov3-vitb16-pretrain-lvd1689m_1024_mp2_gemma-3-270m-it_1xGPU_423791samples_bs64_5000_lr5e-05-0.00512_0905-221819/step_3000"

print(f"Loading model from {checkpoint_path}...")

from models.vision_language_model import VisionLanguageModel

try:
    model = VisionLanguageModel.from_pretrained(checkpoint_path)
    print(f"✓ Model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"✓ Model moved to {device}")
    
    # Try to load tokenizer and processor
    tokenizer_path = os.path.join(checkpoint_path, "tokenizer")
    processor_path = os.path.join(checkpoint_path, "processor")
    
    if os.path.exists(tokenizer_path):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"✓ Tokenizer loaded from checkpoint")
    else:
        print(f"✗ No tokenizer found at {tokenizer_path}")
        print(f"  Using tokenizer from config: {model.cfg.lm_tokenizer}")
        from data.processors import get_tokenizer
        tokenizer = get_tokenizer(
            model.cfg.lm_tokenizer, 
            model.cfg.vlm_extra_tokens, 
            model.cfg.lm_chat_template
        )
        print(f"✓ Tokenizer loaded from HuggingFace")
    
    if os.path.exists(processor_path):
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(processor_path)
        print(f"✓ Image processor loaded from checkpoint")
    else:
        print(f"✗ No processor found at {processor_path}")
        print(f"  Using processor from config: {model.cfg.vit_model_type}")
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(model.cfg.vit_model_type)
        print(f"✓ Image processor loaded from HuggingFace")
    
    # Test a simple forward pass with dummy data
    print("\nTesting forward pass...")
    with torch.no_grad():
        dummy_input = torch.randint(0, 1000, (1, 10)).to(device)
        dummy_images = torch.randn(1, 3, 224, 224).to(device)
        dummy_attention_mask = torch.ones(1, 10).to(device)
        
        logits, loss = model(dummy_input, dummy_images, dummy_attention_mask)
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {logits.shape}")
    
    # Save a simple test result
    output_path = os.path.join(checkpoint_path, "test_results.json")
    test_results = {
        "checkpoint_path": checkpoint_path,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "device": device,
        "forward_test": "passed",
        "output_shape": list(logits.shape),
        "tokenizer_loaded": os.path.exists(tokenizer_path),
        "processor_loaded": os.path.exists(processor_path),
    }
    
    with open(output_path, "w") as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✓ Test results saved to {output_path}")
    print("\nSummary:")
    print(f"  - Model loaded: ✓")
    print(f"  - Forward pass: ✓")  
    print(f"  - Tokenizer: {'✓ (from checkpoint)' if os.path.exists(tokenizer_path) else '✓ (from HuggingFace)'}")
    print(f"  - Processor: {'✓ (from checkpoint)' if os.path.exists(processor_path) else '✓ (from HuggingFace)'}")
    
    # Now try actual evaluation if lmms_eval is available
    try:
        print("\n" + "="*50)
        print("Testing evaluation with lmms_eval...")
        
        # Import evaluation components
        from eval.lmms_eval_wrapper import NanoVLMWrapper
        from lmms_eval import evaluator
        
        # Create wrapper for our model
        wrapper = NanoVLMWrapper(model)
        
        # Run a minimal evaluation
        task_names = ["mmstar"]
        results = evaluator.simple_evaluate(
            model=wrapper,
            tasks=task_names,
            num_fewshot=0,
            batch_size=1,
            device=device,
            limit=2,  # Just 2 samples for testing
        )
        
        if results and "results" in results:
            print(f"✓ Evaluation completed successfully")
            print(f"\nResults preview:")
            for task, metrics in results["results"].items():
                print(f"  {task}:")
                for metric, value in list(metrics.items())[:3]:  # Show first 3 metrics
                    if isinstance(value, float):
                        print(f"    {metric}: {value:.4f}")
                    else:
                        print(f"    {metric}: {value}")
            
            # Save full results
            eval_output_path = os.path.join(checkpoint_path, "eval_test_results.json")
            with open(eval_output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Full evaluation results saved to {eval_output_path}")
        else:
            print("✗ No evaluation results obtained")
            
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)