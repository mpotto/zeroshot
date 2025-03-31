#!/bin/bash

device=2

# CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_template.py --dataset imagenet1k --model RN50 --batch_size 128
# CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_retrieval.py --dataset flickr8k --model ViT-B-32 --batch_size 128 --num_seeds 5
# CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_retrieval.py --dataset mscoco_captions --model ViT-B-32 --batch_size 128 --num_seeds 5

CUDA_VISIBLE_DEVICES=$device python scripts/run_linear_probe.py --dataset dtd --model ViT-B-32 --batch_size 128 --num_seeds 5
CUDA_VISIBLE_DEVICES=$device python scripts/run_linear_probe.py --dataset flowers --model ViT-B-32 --batch_size 128 --num_seeds 5
CUDA_VISIBLE_DEVICES=$device python scripts/run_linear_probe.py --dataset fgvc_aircraft --model ViT-B-32 --batch_size 128 --num_seeds 5
CUDA_VISIBLE_DEVICES=$device python scripts/run_linear_probe.py --dataset sun397 --model ViT-B-32 --batch_size 128 --num_seeds 5

# for seed in {0..4}
# do
#     python scripts/evaluate_imagenet_captions_.py --model ViT-B-32 --device $device --seed $seed --nonuniform class_weight
#     python scripts/evaluate_imagenet_captions_.py --model ViT-B-32 --device $device --seed $seed --nonuniform variance
#     python scripts/evaluate_imagenet_captions_.py --model ViT-B-32 --device $device --seed $seed --nonuniform class_weight_variance
# done