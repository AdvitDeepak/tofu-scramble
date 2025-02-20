import sys
import json
import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

N = 5  # Number of examples to extract
MAX_IDX = 3959  # Maximum index value

def get_examples(target_idx):
    dataset = load_dataset("locuslab/TOFU", "retain99", split="train")
    indices = [(target_idx + i - N // 2) % (MAX_IDX + 1) for i in range(N)]
    examples = [dataset[i] for i in indices]
    return indices, examples

def run_model(model_path, examples, indices):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # from peft import LoraConfig

    # config = LoraConfig.from_pretrained(model_path)
    # print(config)

    tokenizer = AutoTokenizer.from_pretrained("locuslab/tofu_ft_llama2-7b")
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
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
            "question" : question,
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

def main():
    if len(sys.argv) != 3:
        print("Usage: python forget_test.py <model_path> <target_idx>")
        sys.exit(1)

    model_path = sys.argv[1]
    target_idx = int(sys.argv[2])

    print(f"Loading dataset for target index {target_idx}...")
    indices, examples = get_examples(target_idx)

    output_path = f"tests/{target_idx}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load existing results if the file exists
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    if model_path in data: 
        print("Already exists, exiting!")
        exit() 

    print(f"Running model from {model_path} on {len(examples)} examples...")
    responses = run_model(model_path, examples, indices)


    data[model_path] = responses

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
