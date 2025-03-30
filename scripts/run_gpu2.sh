#!/bin/bash

device=2

# CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_template.py --dataset imagenet1k --model RN50 --batch_size 128

for seed in {0..4}
do
    python scripts/evaluate_imagenet_captions_.py --model ViT-B-32 --device $device --seed $seed --nonuniform class_weight
    python scripts/evaluate_imagenet_captions_.py --model ViT-B-32 --device $device --seed $seed --nonuniform variance
    python scripts/evaluate_imagenet_captions_.py --model ViT-B-32 --device $device --seed $seed --nonuniform class_weight_variance
done