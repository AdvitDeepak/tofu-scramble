#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import os
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from UsableXAI_LLM.libs.utils.datatools import init_folder, batchit

import yaml

# Import for influence analysis
from influence_analysis import calculate_influences, analyze_influences, TOFUDataset, estimate_hessian, get_model_config

N = 5  # Number of examples to extract
MAX_IDX = 3959  # Maximum index value

def get_examples(target_idx):
    dataset = load_dataset("locuslab/TOFU", "retain99", split="train")
    indices = [(target_idx + i - N // 2) % (MAX_IDX + 1) for i in range(N)]
    examples = [dataset[i] for i in indices]
    return indices, examples


def run_model(model_path, examples, indices):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("locuslab/tofu_ft_llama2-7b")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, 
    ).to(device)
    
    results = {}

    for i, example in enumerate(tqdm(examples, desc="Processing examples", unit="example")):
        question = example.get("question", "")
        gold_answer = example.get("answer", "")

        inputs = tokenizer(question, return_tensors="pt").to(device)

        # Default model generation
        output = model.generate(**inputs, max_new_tokens=50)
        model_answer = tokenizer.decode(output[0], skip_special_tokens=True)
        model_answer = model_answer.replace(question, "")

        # Beam search decoding
        beam_output = model.generate(**inputs, max_new_tokens=50, num_beams=5, early_stopping=True)
        beam_answer = tokenizer.decode(beam_output[0], skip_special_tokens=True)
        beam_answer = beam_answer.replace(question, "")

        results[indices[i]] = {
            "question": question,
            "gold_answer": gold_answer,
            "model_answer": model_answer,
            "beam_answer": beam_answer
        }

        # Print progress
        print(f"\nQuestion: {question}")
        print(f"Gold Answer: {gold_answer}")
        print(f"Model Answer: {model_answer}")
        print(f"Beam Search Answer: {beam_answer}\n")

    return results

def run_influence_analysis(model_path, target_idx, compute_hessian=False):
    """Run influence analysis to find which examples most influenced the model's prediction."""
    print(f"\n===== Running Influence Analysis for Target {target_idx} =====")
    
    # Set up paths
    model_name = os.path.basename(model_path)
    inf_root = os.path.join("results", model_name)
    
    # Load or compute Hessian approximation
    if compute_hessian or not os.path.exists(os.path.join(inf_root, "layer_svds.pkl")):
        print("Computing Hessian approximation...")
        dataset = TOFUDataset("retain99")
        estimator = estimate_hessian(
            model_path, 
            dataset, 
            batch_size=2,
            max_samples=500
        )
        
        # Save estimator
        os.makedirs(inf_root, exist_ok=True)
        estimator.save_to_disk(inf_root)
    else:
        print("Loading pre-computed Hessian approximation...")
        estimator = None  # Will be loaded in calculate_influences
    
    # Calculate influences
    influences = calculate_influences(
        model_path,
        target_idx,
        estimator
    )
    
    # Analyze and report results
    dataset = TOFUDataset("retain99")
    target_rank = analyze_influences(influences, target_idx, dataset, top_k=10)
    
    return influences, target_rank

def main():
    parser = argparse.ArgumentParser(description="Test and analyze unlearned models")
    parser.add_argument("model_path", type=str, help="Path to the model to test")
    parser.add_argument("target_idx", type=int, help="Target index that was unlearned")
    parser.add_argument("--run_inference", action="store_true", help="Run model inference")
    parser.add_argument("--run_influence", action="store_true", help="Run influence analysis")
    parser.add_argument("--compute_hessian", action="store_true", help="Compute Hessian approximation")
    
    # Check for legacy command-line style
    if len(sys.argv) == 3 and sys.argv[1] and sys.argv[2].isdigit():
        args = parser.parse_args([sys.argv[1], sys.argv[2], "--run_inference"])
    else:
        args = parser.parse_args()
    
    model_path = args.model_path
    target_idx = int(args.target_idx)
    
    output_path = f"tests/{target_idx}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load existing results if the file exists
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    
    # Run model inference
    if args.run_inference:
        if model_path in data: 
            print(f"Model {model_path} already exists in results, skipping inference")
        else:
            print(f"Loading dataset for target index {target_idx}...")
            indices, examples = get_examples(target_idx)
            
            print(f"Running model from {model_path} on {len(examples)} examples...")
            responses = run_model(model_path, examples, indices)
            
            data[model_path] = responses
            
            with open(output_path, "w") as f:
                json.dump(data, f, indent=4)
                
            print(f"\nInference results saved to {output_path}")
    
    # Run influence analysis
    if args.run_influence:
        influence_path = f"tests/influence_{target_idx}.json"
        
        # Load existing results if the file exists
        if os.path.exists(influence_path):
            with open(influence_path, "r") as f:
                influence_data = json.load(f)
        else:
            influence_data = {}
        
        if model_path in influence_data:
            print(f"Model {model_path} already exists in influence results, skipping analysis")
        else:
            influences, target_rank = run_influence_analysis(
                model_path, 
                target_idx,
                compute_hessian=args.compute_hessian
            )
            
            # Store top influences and target rank
            influence_data[model_path] = {
                "target_rank": target_rank,
                "top_influences": [(int(idx), float(score)) for idx, score in influences[:20]]
            }
            
            with open(influence_path, "w") as f:
                json.dump(influence_data, f, indent=4)
                
            print(f"\nInfluence analysis results saved to {influence_path}")
    
    print("\nAll requested operations completed")

if __name__ == "__main__":
    main()