# hf_make.py
# This script creates a dataset of adapter metadata and uploads it to Hugging Face Hub

import os
import json
from huggingface_hub import HfApi
import pandas as pd
import random

data_dir = "models/unlearned-adapters/"
test_dir = "tests/"
hf_org = "advit" 
TOKEN = "" # TODO: add your HF token here

hf_api = HfApi()

if TOKEN == "":
    exit("Uh oh! You need to add your HuggingFace token to this file first!")

# Collect adapter metadata
records = []
for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)
    if os.path.isdir(folder_path):
        print(f"Parsing: {folder}")
        try:
            parts = folder.split("_")
            print(parts)
            loss_type = "_".join(parts[:-3])
            lr = parts[-3]
            target_idx = int(parts[-2])
            epochs = parts[-1]
        except ValueError:
            print(f"Skipping {folder}, invalid naming format.")
            continue
        
        # Upload to Hugging Face
        repo_id = f"{hf_org}/{folder}"
        hf_link = f"https://huggingface.co/{repo_id}"

        try:
            hf_api.repo_info(repo_id, token=TOKEN)  # Try fetching repo info
            print(f"Repository {repo_id} already exists. Skipping upload.")
        except:
            print(f"Creating and uploading {repo_id}...")
            hf_api.create_repo(repo_id, exist_ok=True, token=TOKEN)
            hf_api.upload_folder(folder_path=folder_path, repo_id=repo_id, token=TOKEN)



        # Load test file for alt candidates
        test_file = os.path.join(test_dir, f"{target_idx}.json")
        alt_candidates, target_question = set(), None
        if os.path.exists(test_file):
            with open(test_file, "r") as f:
                test_data = json.load(f)
                if folder_path in test_data:
                    alt_candidates = {entry["question"] for entry in test_data[folder_path].values()}
                    alt_candidates = random.sample(alt_candidates, len(alt_candidates))  # Convert to shuffled list

                    target_question = test_data[folder_path].get(str(target_idx), {}).get("question", None)
        
        records.append([loss_type, lr, target_idx, epochs, hf_link, list(alt_candidates), target_question])


# Create a dataset
dataset_df = pd.DataFrame(records, columns=["loss_type", "learning_rate", "target_idx", "epochs", "hf_link", "alt_candidates", "target_question"])
dataset_df.to_csv("tofu-scramble.csv", index=False)

dataset_repo = f"{hf_org}/tofu-scramble"
hf_api.create_repo(dataset_repo, repo_type="dataset", exist_ok=True, token=TOKEN)
hf_api.upload_file(path_or_fileobj="tofu-scramble.csv", repo_type="dataset", path_in_repo="tofu-scramble.csv", repo_id=dataset_repo, token=TOKEN)
print("Dataset uploaded successfully.")
