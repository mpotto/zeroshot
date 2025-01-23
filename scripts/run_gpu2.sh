#!/bin/bash

device=2
dataset=sun397

CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_complexity.py --dataset $dataset --model nllb-clip-base
CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_complexity.py --dataset $dataset --model RN50
CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_complexity.py --dataset $dataset --model ViT-B-32