#!/usr/bin/env python
# -*- coding: utf-8 -*-
# analyze_tofu_model.py
"""
Analyze the Fine-Tuned TOFU Model (Before Unlearning)

This script analyzes the influence of forget dataset examples on a fine-tuned model
to identify which examples would be most impactful to unlearn.
"""

import os
import sys
import argparse
import json
import torch
import yaml
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import influence analysis components
from influence_analysis import (
    TOFUDataset, 
    estimate_hessian, 
    calculate_influences_from_forget, 
    get_model_config
)
from UsableXAI_LLM.libs.utils.datatools import init_folder

def load_model_configs():
    """Load model configurations from YAML files."""
    # Load forget_solo config
    with open("config/forget_solo.yaml", "r") as f:
        forget_config = yaml.safe_load(f)
    
    # Load model config
    with open("config/model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    
    return forget_config, model_config

def download_hf_model(repo_id, local_dir=None):
    """Download a model from Hugging Face Hub."""
    if local_dir is None:
        local_dir = os.path.join("models", os.path.basename(repo_id))
    
    init_folder(local_dir)
    
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"Model already exists at {local_dir}, skipping download.")
    else:
        print(f"Downloading model from {repo_id} to {local_dir}...")
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
        print(f"Model downloaded to {local_dir}")
    
    return local_dir

def analyze_forget_influence(model_path, num_queries=5, num_forget_samples=None, batch_size=2, max_samples=100):
    """Analyze the influence of forget examples on the fine-tuned model."""
    result_dir = "analysis_results"
    init_folder(result_dir)
    
    # Load datasets
    print("Loading TOFU datasets...")
    retain_dataset = TOFUDataset("retain99")
    forget_dataset = TOFUDataset("forget01")
    
    if num_forget_samples is None:
        num_forget_samples = len(forget_dataset)
    else:
        num_forget_samples = min(num_forget_samples, len(forget_dataset))
    
    print(f"Using {num_queries} retain examples as queries to analyze influence of {num_forget_samples} forget examples")
    
    # Create directory structure
    os.makedirs(os.path.join(result_dir, "tofu_forget_influence"), exist_ok=True)
    
    # Path for saving results
    influence_results_path = os.path.join(result_dir, f"forget_influence_{os.path.basename(model_path)}.pkl")
    
    if os.path.exists(influence_results_path):
        print(f"Loading pre-computed influence results from {influence_results_path}")
        with open(influence_results_path, "rb") as f:
            import pickle
            forget_influence_results = pickle.load(f)
    else:
        # Create safe directory structure
        os.makedirs(os.path.join("results", os.path.basename(model_path)), exist_ok=True)
        
        print(f"Estimating Hessian for model {model_path}...")
        estimator = estimate_hessian(
            model_path, 
            retain_dataset,  # Use retain dataset for Hessian estimation
            model_family="llama2-7b",
            batch_size=batch_size,
            max_samples=max_samples,
            device="cuda"
        )
        
        # Dictionary for storing influence results
        forget_influence_results = {}
        
        print(f"Calculating influence of forget examples on model {model_path}...")
        for query_idx in tqdm(range(num_queries), desc="Processing query examples"):
            try:
                influences = calculate_influences_from_forget(
                    model_path,
                    query_idx,
                    estimator,
                    device="cuda",
                    max_forget_samples=num_forget_samples
                )
                forget_influence_results[query_idx] = influences
            except Exception as e:
                print(f"Error calculating influence for query {query_idx}: {e}")
                continue
        
        # Save results
        with open(influence_results_path, "wb") as f:
            import pickle
            pickle.dump(forget_influence_results, f)
    
    return forget_influence_results

def analyze_results(forget_influence_results):
    """Analyze the influence results to identify the most influential forget examples."""
    # Calculate average influence for each forget example across all queries
    aggregated_influence = {}
    
    for query_idx, influences in forget_influence_results.items():
        for idx, score in influences:
            forget_idx = int(idx)
            if forget_idx not in aggregated_influence:
                aggregated_influence[forget_idx] = {
                    "scores": [],
                    "avg_score": 0,
                    "min_score": float('inf'),
                    "max_score": float('-inf')
                }
            
            aggregated_influence[forget_idx]["scores"].append(score)
            aggregated_influence[forget_idx]["min_score"] = min(aggregated_influence[forget_idx]["min_score"], score)
            aggregated_influence[forget_idx]["max_score"] = max(aggregated_influence[forget_idx]["max_score"], score)
    
    # Calculate average scores
    for forget_idx, data in aggregated_influence.items():
        if data["scores"]:  # Check if there's any data
            data["avg_score"] = sum(data["scores"]) / len(data["scores"])
    
    # Sort by average influence score (highest to lowest)
    sorted_influence = sorted(
        aggregated_influence.items(),
        key=lambda x: abs(x[1]["avg_score"]),
        reverse=True
    )
    
    return sorted_influence

def main():
    parser = argparse.ArgumentParser(description="Analyze the fine-tuned TOFU model to identify influential examples")
    parser.add_argument("--model_path", type=str, default="locuslab/tofu_ft_llama2-7b",
                        help="Path or HF repo ID of the fine-tuned model")
    parser.add_argument("--num_queries", type=int, default=5,
                        help="Number of query examples to use from retain dataset")
    parser.add_argument("--num_forget_samples", type=int, default=None,
                        help="Number of forget samples to analyze (default: all)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for Hessian estimation")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Maximum number of samples for Hessian estimation")
    parser.add_argument("--load_only", action="store_true",
                        help="Only download the model without analysis")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top influential examples to display")
    args = parser.parse_args()
    
    # Load configs
    forget_config, model_config = load_model_configs()
    
    # Download the model
    model_path = download_hf_model(args.model_path)
    
    if args.load_only:
        print(f"Model downloaded to {model_path}")
        return
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Analyze forget example influence
    print("\n===== Analyzing Influence of Forget Examples on Fine-Tuned Model =====")
    forget_influence_results = analyze_forget_influence(
        model_path,
        num_queries=args.num_queries,
        num_forget_samples=args.num_forget_samples,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # Analyze the results
    print("\n===== Identifying Most Influential Forget Examples =====")
    sorted_influence = analyze_results(forget_influence_results)
    
    # Display top influential examples
    print(f"\nTop {args.top_k} Most Influential Forget Examples:")
    forget_dataset = TOFUDataset("forget01")
    for rank, (forget_idx, data) in enumerate(sorted_influence[:args.top_k]):
        if forget_idx < len(forget_dataset):
            example = forget_dataset[forget_idx]
            print(f"Rank {rank+1} | Forget Index {forget_idx} | Avg Influence: {data['avg_score']:.6f}")
            print(f"Min Influence: {data['min_score']:.6f} | Max Influence: {data['max_score']:.6f}")
            print(f"Question: {example['question']}")
            print(f"Answer: {example['answer'][:100]}...")  # Truncate long answers
            print("-" * 80)
    
    # Save results
    result_dir = "analysis_results"
    results = {
        "model_path": args.model_path,
        "num_queries": args.num_queries,
        "num_forget_samples": args.num_forget_samples if args.num_forget_samples else len(forget_dataset),
        "top_influential_examples": [
            {
                "forget_idx": int(idx),
                "avg_score": float(data["avg_score"]),
                "min_score": float(data["min_score"]),
                "max_score": float(data["max_score"])
            }
            for idx, data in sorted_influence[:args.top_k]
        ]
    }
    
    results_path = os.path.join(result_dir, f"tofu_model_influential_examples.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {results_path}")
    
    # Recommend which example would be best to unlearn
    if sorted_influence:
        best_forget_idx = sorted_influence[0][0]
        print(f"\n===== Recommendation =====")
        print(f"The most influential forget example is index {best_forget_idx}.")
        print("This would be the most impactful example to unlearn from the model.")
        
        # Get the example
        if best_forget_idx < len(forget_dataset):
            example = forget_dataset[best_forget_idx]
            print(f"\nQuestion: {example['question']}")
            print(f"Answer: {example['answer']}")

if __name__ == "__main__":
    main()