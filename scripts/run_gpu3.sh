#!/bin/bash

device=3
dataset=dtd

CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_complexity.py --dataset $dataset --model ViT-B-32
CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_complexity.py --dataset $dataset --model nllb-clip-base
CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_complexity.py --dataset $dataset --model RN50