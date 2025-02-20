# TOFU Scramble: Recovering Unlearned Information from TOFU 

> NOTE: this is a fork of the [**TOFU Codebase**](https://locuslab.github.io/tofu), which serves as a benchmark for evaluating unlearning performance of large language models on realistic tasks. The TOFU dataset comprises question-answer pairs based on autobiographies of 200 different authors that do not exist and are completely fictitiously generated by the GPT-4 model. The goal of the task is to unlearn a fine-tuned model on various fractions of the forget set.

## Introduction 

This repository contains the code necessary to create the `advit/tofu-scramble` dataset. This dataset is used to benchmark attack methods which aim to detect specific sequences which were unlearned. 

We take TOFU's finetuned llama2-7b model (`locuslab/tofu_ft_llama2-7b`) and randomly unlearn a specific sequence (Q/A pair) from the TOFU dataset (`locuslab/TOFU`). Then, we create a dataset of LoRA adapters, unlearning hyperparameters, and sets of potential unlearning targets as inputs. The output is the actual sequence which was unlearned. 


## Installation

> Coming soon!

<!-- 

TODO: environment setup
TODO: running download.py
TODO: GPU usage
TODO: command to run 

-->