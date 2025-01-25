#!/bin/bash

device=1

CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_complexity.py --dataset dtd --model ViT-B-32 --batch_size 128
CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_complexity.py --dataset sun397 --model nllb-clip-base --batch_size 128
CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_complexity.py --dataset flowers --model RN50 --batch_size 128

# for seed in {0..9}
# do
#     python scripts/evaluate_imagenet_captions_.py --model nllb-clip-base --device $device --seed $seed
# done
