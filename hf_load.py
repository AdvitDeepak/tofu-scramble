# hf_load.py
# This script downloads or uploads a model from Hugging Face Hub

import os
import argparse
from huggingface_hub import snapshot_download, upload_folder


def download_model(repo_id, local_dir):
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"Model already exists at {local_dir}, skipping download.")
    else:
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
        print(f"Model downloaded locally to {local_dir}")


def upload_model(local_dir, repo_id):
    if os.path.exists(local_dir) and os.listdir(local_dir):
        upload_folder(folder_path=local_dir, repo_id=repo_id)
        print(f"Model uploaded from {local_dir} to {repo_id}")
    else:
        print(f"Local directory {local_dir} does not exist or is empty. Upload aborted.")


def main():
    parser = argparse.ArgumentParser(description="Download or upload models from Hugging Face Hub.")
    parser.add_argument("--upload", action="store_true", help="Upload the model instead of downloading.")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repository ID")
    parser.add_argument("--local_dir", type=str, required=True, help="Local directory for the model")
    
    args = parser.parse_args()

    if args.upload:
        upload_model(args.local_dir, args.repo_id)
    else:
        download_model(args.repo_id, args.local_dir)

if __name__ == "__main__":
    main()
