#!/bin/bash

device=0

CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_complexity.py --dataset imagenet1k --model ViT-B-32 --batch_size 128
# CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_complexity.py --dataset flowers --model nllb-clip-base --batch_size 128
# CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_complexity.py --dataset sun397 --model RN50 --batch_size 128

# for seed in {0..9}
# do
#     python scripts/evaluate_imagenet_captions_.py --model RN50 --device $device --seed $seed
# done

