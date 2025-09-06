#!/usr/bin/env python
"""
Evaluate a saved checkpoint and save metrics to JSON.
"""

import argparse
import json
import os

import torch

from evaluation import cli_evaluate
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
        default="mmstar,mmmu,ocrbench,textvqa",
        help="Comma-separated list of evaluation tasks",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of samples per task (use small number for testing)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save evaluation results JSON (default: checkpoint_path/eval_results.json)",
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

    # Create a minimal eval args namespace that matches what cli_evaluate expects
    eval_args = argparse.Namespace()
    eval_args.model = model
    eval_args.tasks = args.tasks
    eval_args.limit = args.limit
    eval_args.batch_size = args.batch_size
    eval_args.process_with_media = True
    eval_args.device = args.device
    # Add other required fields with defaults
    eval_args.num_fewshot = None
    eval_args.use_cache = None
    eval_args.check_integrity = False
    eval_args.write_out = False
    eval_args.log_samples = False
    eval_args.wandb_log_samples = False
    eval_args.apply_chat_template = False
    eval_args.fewshot_as_multiturn = False
    eval_args.gen_kwargs = None
    eval_args.verbosity = "INFO"
    eval_args.wandb_args = ""
    eval_args.predict_only = False
    eval_args.seed = [1234]
    eval_args.trust_remote_code = False
    eval_args.no_log_wandb = True
    eval_args.output_path = None
    eval_args.model_args = None

    # Run evaluation
    print(f"\nEvaluating on tasks: {args.tasks}")
    print(f"Limit: {args.limit} samples per task")
    eval_results = cli_evaluate(eval_args)

    # Process results
    if eval_results and "results" in eval_results[0]:
        metrics = {}
        for task_name, task_results in eval_results[0]["results"].items():
            task_metrics = {}
            for metric_name, metric_value in task_results.items():
                if isinstance(metric_value, (int, float)):
                    # Clean up metric name (remove suffix after comma)
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
        

if __name__ == "__main__":
    main()