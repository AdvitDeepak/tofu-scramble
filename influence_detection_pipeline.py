#!/usr/bin/env python
# -*- coding: utf-8 -*-
# influence_detection_pipeline.py
"""
TOFU Unlearning Detection Pipeline

This script runs a complete pipeline that:
1. Unlearns a specific example from the TOFU dataset
2. Performs influence analysis on both original and unlearned models
3. Compares the results to identify which example was unlearned
"""

import os
import sys
import random
import subprocess
import argparse
import json
import time
from pathlib import Path
from UsableXAI_LLM.libs.utils.datatools import init_folder

def run_command(command, description=None):
    """Run a shell command and capture its output."""
    if description:
        print(f"\n===== {description} =====")
    
    print(f"Running: {command}")
    
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True
    )
    
    # Capture output in real-time
    stdout_lines = []
    for line in iter(process.stdout.readline, ''):
        print(line, end="")
        stdout_lines.append(line)
    
    # Ensure the process finishes
    process.stdout.close()
    process.wait()
    
    # Capture any stderr output
    stderr = process.stderr.read()
    if stderr:
        print("\n----- STDERR -----")
        print(stderr)
    
    return "".join(stdout_lines), stderr, process.returncode

def run_forget_solo(target_idx, output_dir=None):
    """Run the unlearning process on a target example."""
    command = (
        f"CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 forget_solo.py "
        f"--config-name=forget_solo.yaml target_idx={target_idx}"
    )
    
    if output_dir:
        command += f" save_dir={output_dir}"
    
    stdout, stderr, returncode = run_command(command, description=f"Unlearning Target Index {target_idx}")
    
    # Extract save path from output
    save_path = None
    for line in stdout.split("\n"):
        if line.startswith("SAVE_PATH: "):
            save_path = line.split("SAVE_PATH: ")[1].strip()
            break
    
    if not save_path:
        print("Warning: Could not extract save path from output")
        if output_dir:
            save_path = output_dir
        else:
            # Construct default save path based on forget_solo.yaml pattern
            save_path = f"models/unlearned-adapters/grad_diff_1e-5_{target_idx}_120"
    
    return save_path, returncode == 0

def run_influence_analysis(model_path, target_idx, compute_hessian=False):
    """Run influence analysis on a model for a target example."""
    command = (
        f"python forget_test_with_influence.py {model_path} {target_idx} "
        f"--run_influence"
    )
    
    if compute_hessian:
        command += " --compute_hessian"
    
    stdout, stderr, returncode = run_command(
        command, 
        description=f"Running Influence Analysis on {os.path.basename(model_path)}"
    )
    
    influence_path = f"tests/influence_{target_idx}.json"
    return influence_path if os.path.exists(influence_path) else None, returncode == 0

def run_comparison(original_results, unlearned_results, target_idx, output="comparison_results.json"):
    """Compare influence results to identify the unlearned datapoint."""
    command = (
        f"python compare_influences.py "
        f"--original_path {original_results} "
        f"--unlearned_path {unlearned_results} "
        f"--target_idx {target_idx} "
        f"--output {output}"
    )
    
    stdout, stderr, returncode = run_command(
        command, 
        description="Comparing Influence Results"
    )
    
    return output if os.path.exists(output) else None, returncode == 0

def load_json(path):
    """Load a JSON file."""
    if not os.path.exists(path):
        return None
    
    with open(path, "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Run the complete unlearning detection pipeline")
    parser.add_argument("--target_idx", type=int, help="Target index to unlearn (random if not specified)")
    parser.add_argument("--original_model", type=str, default="models/tofu_ft_llama2-7b", 
                        help="Path to the original model")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Output directory for unlearned model (default: auto-generated)")
    parser.add_argument("--skip_unlearning", action="store_true", 
                        help="Skip the unlearning step (use with --unlearned_model)")
    parser.add_argument("--unlearned_model", type=str, 
                        help="Path to pre-unlearned model (use with --skip_unlearning)")
    parser.add_argument("--skip_influence", action="store_true", 
                        help="Skip running influence analysis (use with --original_results and --unlearned_results)")
    parser.add_argument("--original_results", type=str, 
                        help="Path to original model influence results (use with --skip_influence)")
    parser.add_argument("--unlearned_results", type=str, 
                        help="Path to unlearned model influence results (use with --skip_influence)")
    parser.add_argument("--compute_hessian", action="store_true", 
                        help="Compute Hessian approximation (slower but more accurate)")
    args = parser.parse_args()
    
    # Create a timestamp for this run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Set output directory if not specified
    if not args.output_dir:
        args.output_dir = f"models/unlearned-{timestamp}"
    
    # Create necessary directories
    os.makedirs("tests", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Choose a random target if not specified
    target_idx = args.target_idx
    if target_idx is None:
        target_idx = random.randint(0, 3959)
        print(f"Randomly selected target index: {target_idx}")
    
    # Create a run directory to store all results
    run_dir = f"runs/{timestamp}-target-{target_idx}"
    init_folder(run_dir)
    
    # Save run configuration
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Step 1: Unlearn the target example
    unlearned_model = args.unlearned_model
    if not args.skip_unlearning:
        print("\n==== Step 1: Unlearning Target Example ====")
        unlearned_model, success = run_forget_solo(target_idx, args.output_dir)
        if not success or not unlearned_model:
            print("⚠️ Unlearning process failed or didn't return a model path")
            return 1
    else:
        if not unlearned_model:
            print("⚠️ --skip_unlearning requires --unlearned_model")
            return 1
        print(f"\n==== Step 1: Using Pre-unlearned Model at {unlearned_model} ====")
    
    # Step 2: Run influence analysis on both models
    original_results = args.original_results
    unlearned_results = args.unlearned_results
    
    if not args.skip_influence:
        print("\n==== Step 2: Running Influence Analysis ====")
        
        print("\nAnalyzing original model...")
        original_results, success = run_influence_analysis(
            args.original_model, 
            target_idx,
            compute_hessian=args.compute_hessian
        )
        if not success or not original_results:
            print("⚠️ Influence analysis failed on original model")
            return 1
        
        print("\nAnalyzing unlearned model...")
        unlearned_results, success = run_influence_analysis(
            unlearned_model, 
            target_idx,
            compute_hessian=args.compute_hessian
        )
        if not success or not unlearned_results:
            print("⚠️ Influence analysis failed on unlearned model")
            return 1
    else:
        if not original_results or not unlearned_results:
            print("⚠️ --skip_influence requires both --original_results and --unlearned_results")
            return 1
        print(f"\n==== Step 2: Using Pre-computed Influence Results ====")
        print(f"Original results: {original_results}")
        print(f"Unlearned results: {unlearned_results}")
    
    # Step 3: Compare results to identify the unlearned datapoint
    print("\n==== Step 3: Comparing Influence Results ====")
    comparison_output = os.path.join(run_dir, "comparison_results.json")
    comparison_path, success = run_comparison(
        original_results,
        unlearned_results,
        target_idx,
        output=comparison_output
    )
    
    if not success or not comparison_path:
        print("⚠️ Comparison failed")
        return 1
    
    # Load and display the final results
    comparison_results = load_json(comparison_path)
    if not comparison_results:
        print("⚠️ Could not load comparison results")
        return 1
    
    predicted_idx = comparison_results["predicted_unlearned_idx"]
    is_correct = comparison_results["prediction_correct"]
    
    print("\n===== FINAL RESULTS =====")
    print(f"Target Index: {target_idx}")
    print(f"Predicted Index: {predicted_idx}")
    print(f"Correct: {is_correct}")
    
    if is_correct:
        print("\n✅ SUCCESS: The influence analysis correctly identified the unlearned datapoint!")
    else:
        print("\n❌ FAILURE: The influence analysis did not correctly identify the unlearned datapoint.")
    
    # Copy plot to run directory
    if "plot_path" in comparison_results and os.path.exists(comparison_results["plot_path"]):
        plot_name = os.path.basename(comparison_results["plot_path"])
        os.system(f"cp {comparison_results['plot_path']} {run_dir}/{plot_name}")
        print(f"\nPlot saved to {run_dir}/{plot_name}")
    
    print(f"\nAll results saved in {run_dir}")
    
    return 0 if is_correct else 1

if __name__ == "__main__":
    sys.exit(main())