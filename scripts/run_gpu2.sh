#!/bin/bash

device=2

CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_complexity.py --dataset sun397 --model ViT-B-32 --batch_size 128
CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_complexity.py --dataset dtd --model nllb-clip-base --batch_size 128
CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_complexity.py --dataset dtd --model RN50 --batch_size 128

# for seed in {0..9}
# do
#     python scripts/evaluate_imagenet_captions_.py --model ViT-B-32 --device $device --seed $seed
# done