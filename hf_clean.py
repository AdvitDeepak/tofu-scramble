import os
import json

def update_adapter_configs(root_dir, new_model_name):
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        config_path = os.path.join(folder_path, "adapter_config.json")
        
        if os.path.isdir(folder_path) and os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                
                if config.get("base_model_name_or_path") != new_model_name:
                    config["base_model_name_or_path"] = new_model_name
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(config, f, indent=4)
                    print(f"Updated: {config_path}")
                else:
                    print(f"No change needed: {config_path}")
            except Exception as e:
                print(f"Error processing {config_path}: {e}")

if __name__ == "__main__":
    root_directory = "models/unlearned-adapters"
    new_model_name = "locuslab/tofu_ft_llama2-7b"
    update_adapter_configs(root_directory, new_model_name)
