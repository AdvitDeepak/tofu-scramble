# [**TOFU Scramble**](https://huggingface.co/datasets/advit/tofu-scramble): Recovering Unlearned Information from TOFU 

> NOTE: this is a fork of the [**TOFU Codebase**](https://locuslab.github.io/tofu), which serves as a benchmark for evaluating unlearning performance of large language models on realistic tasks. The TOFU dataset comprises question-answer pairs based on autobiographies of 200 different authors that do not exist and are completely fictitiously generated by the GPT-4 model. The goal of the task is to unlearn a fine-tuned model on various fractions of the forget set.



## Introduction 

This repository contains the code necessary to create the `advit/tofu-scramble` dataset. This dataset is used to benchmark attack methods which aim to detect specific sequences which were unlearned. 

We take TOFU's finetuned llama2-7b model (`locuslab/tofu_ft_llama2-7b`) and randomly unlearn a specific sequence (Q/A pair) from the TOFU dataset (`locuslab/TOFU`). Then, we create a dataset of LoRA adapters, unlearning hyperparameters, and sets of potential unlearning targets as inputs. The output is the actual sequence which was unlearned. 


## Installation

> Proper instructions coming soon, this is just a rough version for now!

* First, setup the environment using `environment.yml` 

* Next, run `python hf_load.py --repo_id locuslab/tofu_ft_llama2-7b --local_dir models/tofu_ft_llama2-7b`. Confirm that the files (i.e `config.json`, `model-XXX-of-XXX.safetensors` are present within `models/tofu_ft_llama2-7b`)

* Then, ensure GPU configuration (will need 2 GPUs, w/ 48 GB vRAM each) 

* Finally, run `python main.py` as many times as desired. Adapters will be stored in `models/unlearned-adapters` and test `.json`s will be stored in `tests/`. 

* To push to Huggingface, run `python hf_clean.py` (to change some local paths) and then `python hf_make.py`.
