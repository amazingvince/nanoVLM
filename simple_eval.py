#!/usr/bin/env python
"""
Simple evaluation script for saved checkpoints.
"""

import argparse
import json
import os

import torch

from models.vision_language_model import VisionLanguageModel


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved VLM checkpoint")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint directory",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="mmstar",
        help="Comma-separated list of evaluation tasks",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of samples per task",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save evaluation results JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation",
    )

    args = parser.parse_args()

    # Load the model
    print(f"Loading model from {args.checkpoint_path}...")
    model = VisionLanguageModel.from_pretrained(args.checkpoint_path)
    model.to(args.device)
    model.eval()
    print(f"Model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Import and run evaluation
    from evaluation import cli_evaluate

    # Create eval args for cli_evaluate
    class EvalArgs:
        def __init__(self):
            self.model = model
            self.tasks = args.tasks
            self.limit = args.limit
            self.batch_size = args.batch_size
            self.process_with_media = True
            self.device = args.device
            self.num_fewshot = None
            self.use_cache = None
            self.check_integrity = False
            self.write_out = False
            self.log_samples = False
            self.wandb_log_samples = False
            self.apply_chat_template = False
            self.fewshot_as_multiturn = False
            self.gen_kwargs = None
            self.verbosity = "INFO"
            self.wandb_args = ""
            self.predict_only = False
            self.seed = [1234]
            self.trust_remote_code = False
            self.no_log_wandb = True
            self.output_path = None
            self.model_args = None
            self.cache_requests = None
            self.log_samples_suffix = None
            self.system_instruction = None
            self.show_config = False
            self.include_path = None
            self.timezone = None
            self.hf_hub_log_args = None
            self.max_batch_size = None
            self.config = None
    
    eval_args = EvalArgs()

    # Run evaluation
    print(f"\nEvaluating on tasks: {args.tasks}")
    print(f"Limit: {args.limit} samples per task")
    print(f"Batch size: {args.batch_size}")
    print("-" * 50)
    
    try:
        eval_results = cli_evaluate(eval_args)
        
        # Process results
        if eval_results and "results" in eval_results[0]:
            metrics = {}
            for task_name, task_results in eval_results[0]["results"].items():
                task_metrics = {}
                for metric_name, metric_value in task_results.items():
                    if isinstance(metric_value, (int, float)):
                        # Clean up metric name
                        clean_metric_name = metric_name.split(",")[0]
                        task_metrics[clean_metric_name] = metric_value
                metrics[task_name] = task_metrics

            # Save results
            output_path = args.output_json
            if output_path is None:
                output_path = os.path.join(args.checkpoint_path, "eval_results.json")
            
            output_data = {
                "checkpoint_path": args.checkpoint_path,
                "tasks": args.tasks,
                "limit": args.limit,
                "batch_size": args.batch_size,
                "metrics": metrics,
            }
            
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\nEvaluation results saved to: {output_path}")
            
            # Print summary
            print("\n=== Evaluation Results ===")
            for task, task_metrics in metrics.items():
                print(f"\n{task}:")
                for metric, value in task_metrics.items():
                    print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
        else:
            print("No evaluation results obtained")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        

if __name__ == "__main__":
    main()