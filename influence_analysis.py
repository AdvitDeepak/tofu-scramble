#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Influence Analysis for TOFU Dataset

This script computes the influence of training examples on a model's predictions
using the EK-FAC approximation. It helps identify which examples most affected
a model's behavior when they were unlearned.
"""

import os
import torch
import random
import argparse
import tqdm
import pickle as pkl
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import yaml

# Import the influence function implementation
from UsableXAI_LLM.libs.core.generator import Generator, zero_grad
from UsableXAI_LLM.libs.core.hooks import MLPHookController
from UsableXAI_LLM.libs.core.EKFAC_influence import CovarianceEstimator, InfluenceEstimator
from UsableXAI_LLM.libs.utils.datatools import batchit, CorpusSearchIndex

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

class TOFUDataset(Dataset):
    """Dataset wrapper for TOFU dataset with QA pairs."""
    
    def __init__(self, split="retain99", subset_size=None):
        """Initialize the dataset.
        
        Args:
            split: The dataset split to load ("retain99" or "forget01")
            subset_size: Optional number of examples to use (for testing)
        """
        self.data = load_dataset("locuslab/TOFU", split, split="train")
        if subset_size is not None:
            self.data = self.data.select(range(min(subset_size, len(self.data))))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "question": item["question"],
            "answer": item["answer"],
            "index": idx
        }

def get_model_config(model_family):
    """Load model configuration from YAML file."""
    with open("config/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config.get(model_family, {})

def prepare_text_for_model(question, answer=None, model_family="llama2-7b"):
    """Format the question and optional answer for the model based on model family."""
    model_config = get_model_config(model_family)
    question_start_tag = model_config.get("question_start_tag", "[INST] ")
    question_end_tag = model_config.get("question_end_tag", " [/INST]")
    answer_tag = model_config.get("answer_tag", "")
    
    if answer:
        return f"{question_start_tag}{question}{question_end_tag}{answer_tag}{answer}"
    else:
        return f"{question_start_tag}{question}{question_end_tag}{answer_tag}"
    
def compute_LM_loss(ids, masks, probs):
    """Compute language modeling loss."""
    bs, ts = ids.shape
    probs = probs[:, :-1, :].reshape(bs * (ts - 1), -1)
    probs = probs[torch.arange(bs * (ts - 1)), ids[:, 1:].flatten()].reshape(bs, ts - 1)
    return -(masks[:, :-1] * torch.log2(probs)).sum(axis=1)

def compute_pseudo_loss(masks, logits):
    """Compute loss based on the model's own predictions."""
    bs, ts = masks.shape
    ids = logits.argmax(dim=-1)  # assuming pseudo labels are greedy-search generated    
    probs = torch.softmax(logits, -1).reshape(bs * ts, -1)
    probs = probs[torch.arange(bs * ts), ids.flatten()].reshape(bs, ts)
    return -(masks * torch.log2(probs)).sum(axis=1)

def get_sample_indices(num_samples, num_neg, i):
    """Get indices for the target and negative samples."""
    indices = list(range(num_samples))
    indices.remove(i)
    neg_indices = random.sample(indices, num_neg)
    return [i] + neg_indices

def estimate_hessian(model_path, dataset, model_family="llama2-7b", batch_size=2, max_samples=500, device="cuda"):
    """Estimate the Hessian approximation using EK-FAC."""
    print("Loading model for Hessian estimation...")
    generator = Generator(model_path, device=device)
    
    # Select appropriate hook controller based on model architecture
    if "llama" in model_path.lower() or "llama" in model_family.lower():
        hooker = MLPHookController.LLaMA(generator._model)
    else:
        hooker = MLPHookController.GPT2(generator._model)
    
    # Initialize covariance estimator
    estimator = CovarianceEstimator()
    
    # Estimate covariance matrices
    print("Estimating covariance matrices (S and A)...")
    bar_cov = tqdm.tqdm(total=min(len(dataset), max_samples), desc="Estimating Covariance")
    
    for i, batch in enumerate(batchit(dataset, batch_size)):
        if i >= max_samples // batch_size:
            break
            
        texts = [prepare_text_for_model(item["question"], item["answer"], model_family) for item in batch]
        
        zero_grad(generator._model)
        inputs, outputs = generator.forward(texts)
        losses = compute_pseudo_loss(inputs["attention_mask"], outputs.logits)
        
        for loss in losses:
            loss.backward(retain_graph=True)
            
        with torch.no_grad():
            estimator.update_cov(hooker.collect_states(),
                                inputs["attention_mask"].to(device))
        
        bar_cov.update(len(texts))
    
    # Calculate eigendecomposition
    print("Calculating eigendecomposition...")
    estimator.calculate_eigenvalues_and_vectors()
    
    # Estimate Lambda values
    print("Estimating Lambda values...")
    batch_size = 1  # Reduce batch size for more accurate Lambda estimation
    bar_lambda = tqdm.tqdm(total=min(len(dataset), max_samples), desc="Estimating Lambda")
    
    for i, batch in enumerate(batchit(dataset, batch_size)):
        if i >= max_samples:
            break
            
        texts = [prepare_text_for_model(item["question"], item["answer"]) for item in batch]
        
        zero_grad(generator._model)
        inputs, outputs = generator.forward(texts)
        losses = compute_pseudo_loss(inputs["attention_mask"], outputs.logits)
        
        for loss in losses:
            loss.backward(retain_graph=True)
            
        with torch.no_grad():
            estimator.update_lambdas(hooker.collect_states(),
                                    inputs["attention_mask"].to(device))
        
        bar_lambda.update(len(texts))
    
    return estimator

def calculate_influences_from_forget(model_path, query_idx, estimator, device="cuda", max_forget_samples=None):
    """Calculate influence scores of forget examples on a query example from retain dataset."""
    print(f"Calculating influence of forget examples on retain query index {query_idx}...")
    
    # Load datasets - query from retain, training examples from forget
    query_dataset = TOFUDataset("retain99")
    forget_dataset = TOFUDataset("forget01")

    # Limit the number of forget samples if specified
    if max_forget_samples is not None:
        forget_samples = min(len(forget_dataset), max_forget_samples)
    else:
        forget_samples = len(forget_dataset)
        
    print(f"Using {forget_samples} examples from forget dataset")
    
    # Load model and hook
    generator = Generator(model_path, device=device)
    if "llama" in model_path.lower():
        hooker = MLPHookController.LLaMA(generator._model)
    else:
        hooker = MLPHookController.GPT2(generator._model)
    
    # Initialize influence estimator
    inf_root = os.path.join("results", os.path.basename(model_path))
    os.makedirs(inf_root, exist_ok=True)
    estimator.save_to_disk(inf_root)
    inf_estimator = InfluenceEstimator.load_from_disk(inf_root)
    
    # Get query example from retain dataset
    query_example = query_dataset[query_idx]
    query_text = prepare_text_for_model(query_example["question"], query_example["answer"], "llama2-7b")
    
    # Forward pass on query example
    tokens = generator._tokenizer.tokenize(query_text)
    ids = torch.tensor([generator._tokenizer.convert_tokens_to_ids(tokens)]).to(device)
    outputs = generator._model(input_ids=ids)
    probs = torch.softmax(outputs.logits, dim=-1)
    out_mask = torch.ones_like(ids).to(device)
    
    # Calculate loss and gradients for query
    query_loss = compute_LM_loss(ids, out_mask, probs)[0]
    query_loss.backward()
    
    with torch.no_grad():
        query_grads = hooker.collect_weight_grads()
        query_grads = {layer: grad.clone() for layer, (_, grad) in query_grads.items()}
    zero_grad(generator._model)
    
    # Calculate HVP for the query
    query_hvps = inf_estimator.calculate_hvp(query_grads)
    
    # Calculate influence for each forget example
    influences = []
    bar = tqdm.tqdm(total=forget_samples, desc=f"Calculating Influence for Query {query_idx}")
    
    for j in range(forget_samples):
        # Get training example from forget dataset
        train_example = forget_dataset[j]
        train_text = prepare_text_for_model(train_example["question"], train_example["answer"], "llama2-7b")
        
        # Forward pass
        inputs, outputs = generator.forward([train_text])
        losses = compute_pseudo_loss(inputs["attention_mask"], outputs.logits)
        
        for loss in losses:
            loss.backward(retain_graph=True)
        
        # Calculate influence
        with torch.no_grad():
            grads = hooker.collect_weight_grads()
            inf = inf_estimator.calculate_total_influence(query_hvps, grads)
            # Store forget index and influence score
            influences.append((j, float(inf.cpu().numpy())))
        
        zero_grad(generator._model)
        bar.update(1)
    
    # Save influence scores
    results_dir = os.path.join("results", os.path.basename(model_path), "forget_influences")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, f"query_{query_idx}_influences.pkl"), "wb") as f:
        pkl.dump(influences, f)
    
    # Return sorted influences
    return sorted(influences, key=lambda x: x[1], reverse=True)

def analyze_influences(influences, target_idx, dataset, top_k=10):
    """Analyze and report the most influential examples."""
    print(f"\n===== Top {top_k} Most Influential Examples for Target {target_idx} =====")
    
    for rank, (idx, score) in enumerate(influences[:top_k]):
        example = dataset[idx]
        print(f"Rank {rank+1} | Index {idx} | Influence Score: {score:.6f}")
        print(f"Question: {example['question']}")
        print(f"Answer: {example['answer']}")
        print("-" * 80)
    
    # Check where the target ranks in influence scores
    target_rank = next((i+1 for i, (idx, _) in enumerate(influences) if idx == target_idx), "Not found")
    print(f"\nTarget example {target_idx} ranks {target_rank} in influence scores")
    
    return target_rank

def main():
    parser = argparse.ArgumentParser(description="Calculate influence scores for TOFU dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model to analyze")
    parser.add_argument("--target_idx", type=int, required=True, help="Index of target example to analyze")
    parser.add_argument("--compute_hessian", action="store_true", help="Compute Hessian approximation (skip if already computed)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for Hessian estimation")
    parser.add_argument("--max_samples", type=int, default=500, help="Maximum number of samples for Hessian estimation")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top influential examples to report")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set up paths
    model_name = os.path.basename(args.model_path)
    inf_root = os.path.join("results", model_name)
    
    # Load or compute Hessian approximation
    if args.compute_hessian or not os.path.exists(os.path.join(inf_root, "layer_svds.pkl")):
        print("Computing Hessian approximation...")
        dataset = TOFUDataset("retain99")
        estimator = estimate_hessian(
            args.model_path, 
            dataset, 
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            device=device
        )
        
        # Save estimator
        os.makedirs(inf_root, exist_ok=True)
        estimator.save_to_disk(inf_root)
    else:
        print("Loading pre-computed Hessian approximation...")
        estimator = None  # Will be loaded in calculate_influences
    
    # Calculate influences
    influences = calculate_influences(
        args.model_path,
        args.target_idx,
        estimator,
        device=device
    )
    
    # Analyze and report results
    dataset = TOFUDataset("retain99")
    target_rank = analyze_influences(influences, args.target_idx, dataset, top_k=args.top_k)
    
    # Save summary
    summary = {
        "model_path": args.model_path,
        "target_idx": args.target_idx,
        "target_rank": target_rank,
        "top_influences": [(idx, float(score)) for idx, score in influences[:args.top_k]]
    }
    
    summary_path = os.path.join(inf_root, f"target_{args.target_idx}_summary.pkl")
    with open(summary_path, "wb") as f:
        pkl.dump(summary, f)
    
    print(f"Analysis complete. Summary saved to {summary_path}")

if __name__ == "__main__":
    main()