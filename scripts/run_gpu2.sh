#!/bin/bash

device=2
dataset=sun397

CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_template.py --dataset $dataset --model ViT-B-32 --batch_size 128
# CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_template.py --dataset $dataset --model nllb-clip-base --batch_size 128
# CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_template.py --dataset $dataset --model RN50 --batch_size 128