# main.py

import random
import subprocess
import re


def run_forget_solo(target_idx):
    formatted_target_idx = f"[{','.join(map(str, target_idx))}]"  # Format as [0,1,2]

    command = (
        f"CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 forget_solo.py "
        f"--config-name=forget_solo.yaml target_idx={formatted_target_idx}"
    )

    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    save_path = None
    log_output = []

    # Read stdout and stderr in real-time
    for line in iter(process.stdout.readline, ''):
        print(line, end="")  # Print line immediately
        log_output.append(line)  # Store output
        if line.startswith("SAVE_PATH: "):
            save_path = line.split("SAVE_PATH: ")[1].strip()

    # Ensure the process finishes
    process.stdout.close()
    process.wait()

    # Read and print any remaining stderr output
    stderr_output = process.stderr.read()
    if stderr_output:
        print("\n=== STDERR OUTPUT ===\n", stderr_output)
        log_output.append(stderr_output)

    return save_path, log_output  # Return the extracted path and full logs


def run_forget_test(model_path, target_idx):
    formatted_target_idx = f"{','.join(map(str, target_idx))}"  # Format as [0,1,2]

    command = ["python", "forget_test.py", model_path, str(formatted_target_idx)]
    subprocess.run(command, timeout=None)


def main():
    #target_idx = random.randint(0, 3959)
    #print(f"Chosen idx: {target_idx}")

    target_idx = [0,1,2]
    target_idx=[2,1,0]
    
    save_path, log_output = run_forget_solo(target_idx)
    print(f"Finished model unlearning!")
    print(f"Save Path: {save_path}")
    #print(log_output)

    #check = input("Continue w/ Testing? (y/n): ")
    #if check != "y": exit()

    #save_path = "models/unlearned-adapters/grad_diff_1e-05_10_0-1-2"
    run_forget_test("models/tofu_ft_llama2-7b", target_idx)
    run_forget_test(save_path, target_idx)


if __name__ == "__main__":
    main()
