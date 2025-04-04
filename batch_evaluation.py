#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch Evaluation of Influence-Based Unlearning Detection

This script runs the unlearning detection pipeline on multiple examples 
and evaluates the accuracy of the influence-based detection method.
"""

import os
import sys
import random
import argparse
import json
import time
import subprocess
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def run_detection(target_idx, skip_unlearning=False, unlearned_model=None, compute_hessian=False):
    """Run the unlearning detection pipeline for a single target."""
    command = [
        "python", "influence_detection_pipeline.py",
        "--target_idx", str(target_idx),
        "--use_forget_dataset"  # Add this flag to use the forget dataset
    ]
    
    if skip_unlearning and unlearned_model:
        command.extend(["--skip_unlearning", "--unlearned_model", unlearned_model])
    
    if compute_hessian:
        command.append("--compute_hessian")
        
    # Add default model paths based on config
    command.extend(["--original_model", "models/tofu_ft_llama2-7b"])
        
    # Set output directory based on config pattern
    output_dir = f"models/unlearned-adapters/grad_diff_1e-5_{target_idx}_120"
    command.extend(["--output_dir", output_dir])
    
    # Run the command
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Capture output
    stdout, stderr = process.communicate()
    
    # Process output to determine if detection was successful
    is_correct = "SUCCESS:" in stdout
    
    # Extract the run directory path
    run_dir = None
    for line in stdout.split("\n"):
        if "All results saved in" in line:
            run_dir = line.split("All results saved in ")[1].strip()
            break
    
    return {
        "target_idx": target_idx,
        "success": is_correct,
        "run_dir": run_dir,
        "return_code": process.returncode
    }

def load_results(run_dir):
    """Load comparison results from a run directory."""
    comparison_path = os.path.join(run_dir, "comparison_results.json")
    if not os.path.exists(comparison_path):
        return None
    
    with open(comparison_path, "r") as f:
        data = json.load(f)
    
    # Handle both old and new result formats
    if "predicted_forget_idx" in data:
        data["predicted_idx"] = data["predicted_forget_idx"]
        
    return data

def analyze_batch_results(results):
    """Analyze the results of batch evaluation."""
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    accuracy = successful / total if total > 0 else 0
    
    # Count how often the target appears in the top-k predictions
    top_k_counts = {k: 0 for k in [1, 3, 5, 10, 20]}
    
    for result in results:
        if not result["run_dir"]:
            continue
        
        comparison_results = load_results(result["run_dir"])
        if not comparison_results:
            continue
        
        target_idx = comparison_results["target_idx"]
        
        # Check if target appears in top-k predictions
        # Format may vary depending on whether we used forget dataset
        if "top_candidates" in comparison_results:
            # New format with forget dataset
            top_changes = comparison_results["top_candidates"]
            top_indices = [item[0] for item in top_changes]  # Extract indices
        else:
            # Old format
            top_changes = comparison_results.get("top_changes", [])
            top_indices = [idx for idx, _ in top_changes]
        
        for k in top_k_counts.keys():
            if k <= len(top_indices) and target_idx in top_indices[:k]:
                top_k_counts[k] += 1
    
    # Calculate top-k recall
    top_k_recall = {k: count / total if total > 0 else 0 
                   for k, count in top_k_counts.items()}
    
    return {
        "total": total,
        "successful": successful,
        "accuracy": accuracy,
        "top_k_recall": top_k_recall
    }

def plot_results(analysis, output_path="batch_results.png"):
    """Plot the batch evaluation results."""
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot accuracy
    ax1.bar(["Accuracy"], [analysis["accuracy"]], color="blue", width=0.4)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Detection Accuracy: {analysis['accuracy']:.2f}")
    ax1.text(0, analysis["accuracy"] / 2, f"{analysis['successful']}/{analysis['total']}", 
            ha='center', va='center', color='white', fontweight='bold')
    
    # Plot top-k recall
    k_values = list(analysis["top_k_recall"].keys())
    recall_values = list(analysis["top_k_recall"].values())
    
    ax2.bar(k_values, recall_values, color="green")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Recall@k")
    ax2.set_title("Top-k Recall")
    
    # Add numbers on bars
    for i, v in enumerate(recall_values):
        ax2.text(i, v / 2, f"{v:.2f}", ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Results plot saved to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Batch evaluation of unlearning detection")
    parser.add_argument("--num_samples", type=int, default=10, 
                        help="Number of samples to evaluate")
    parser.add_argument("--target_indices", type=str, 
                        help="Comma-separated list of target indices to evaluate (overrides num_samples)")
    parser.add_argument("--compute_hessian", action="store_true", 
                        help="Compute Hessian approximation (slower but more accurate)")
    parser.add_argument("--skip_unlearning", action="store_true", 
                        help="Skip unlearning step and use pre-unlearned models")
    parser.add_argument("--unlearned_models_dir", type=str, 
                        help="Directory containing pre-unlearned models (required if skip_unlearning is True)")
    parser.add_argument("--output", type=str, default="batch_evaluation_results.json", 
                        help="Path to save batch evaluation results")
    args = parser.parse_args()
    
    # Create timestamp for this batch evaluation
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    batch_dir = f"batch_evaluation_{timestamp}"
    os.makedirs(batch_dir, exist_ok=True)
    
    # Determine target indices
    if args.target_indices:
        target_indices = [int(idx) for idx in args.target_indices.split(",")]
    else:
        target_indices = random.sample(range(3960), args.num_samples)
    
    # Get unlearned models if skipping unlearning
    unlearned_models = {}
    if args.skip_unlearning:
        if not args.unlearned_models_dir:
            print("⚠️ --skip_unlearning requires --unlearned_models_dir")
            return 1
        
        # Map target indices to unlearned model paths
        for target_idx in target_indices:
            potential_path = os.path.join(args.unlearned_models_dir, f"target_{target_idx}")
            if os.path.exists(potential_path):
                unlearned_models[target_idx] = potential_path
        
        # Filter target indices to only those with unlearned models
        missing_models = set(target_indices) - set(unlearned_models.keys())
        if missing_models:
            print(f"⚠️ Missing unlearned models for {len(missing_models)} target indices")
            print(f"Continuing with {len(unlearned_models)} targets")
            target_indices = list(unlearned_models.keys())
    
    # Save configuration
    with open(os.path.join(batch_dir, "config.json"), "w") as f:
        json.dump({
            "timestamp": timestamp,
            "num_samples": len(target_indices),
            "target_indices": target_indices,
            "compute_hessian": args.compute_hessian,
            "skip_unlearning": args.skip_unlearning,
            "unlearned_models_dir": args.unlearned_models_dir
        }, f, indent=4)
    
    # Run detection for each target
    results = []
    for target_idx in tqdm(target_indices, desc="Evaluating targets"):
        unlearned_model = unlearned_models.get(target_idx) if args.skip_unlearning else None
        result = run_detection(
            target_idx, 
            skip_unlearning=args.skip_unlearning,
            unlearned_model=unlearned_model,
            compute_hessian=args.compute_hessian
        )
        results.append(result)
        
        # Save incremental results
        with open(os.path.join(batch_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
    
    # Analyze results
    analysis = analyze_batch_results(results)
    
    # Create plots
    plot_path = os.path.join(batch_dir, "results_plot.png")
    plot_results(analysis, output_path=plot_path)
    
    # Save final results
    final_results = {
        "timestamp": timestamp,
        "config": {
            "num_samples": len(target_indices),
            "target_indices": target_indices,
            "compute_hessian": args.compute_hessian,
            "skip_unlearning": args.skip_unlearning,
        },
        "individual_results": results,
        "analysis": analysis
    }
    
    with open(os.path.join(batch_dir, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=4)
    
    with open(args.output, "w") as f:
        json.dump(final_results, f, indent=4)
    
    # Print summary
    print("\n===== BATCH EVALUATION SUMMARY =====")
    print(f"Total targets evaluated: {analysis['total']}")
    print(f"Successful detections: {analysis['successful']}")
    print(f"Accuracy: {analysis['accuracy']:.4f}")
    print("\nTop-k Recall:")
    for k, recall in analysis["top_k_recall"].items():
        print(f"  Recall@{k}: {recall:.4f}")
    
    print(f"\nResults saved to {batch_dir}/final_results.json")
    print(f"Plot saved to {plot_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())