#!/bin/bash

device=1

# CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_template.py --dataset imagenet1k --model nllb-clip-base --batch_size 128
# CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_retrieval.py --dataset mscoco_captions --model RN50 --batch_size 128 --num_seeds 5
# CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_retrieval.py --dataset flickr8k --model RN50 --batch_size 128 --num_seeds 5

CUDA_VISIBLE_DEVICES=$device python scripts/run_linear_probe.py --dataset dtd --model RN50 --batch_size 128 --num_seeds 5
CUDA_VISIBLE_DEVICES=$device python scripts/run_linear_probe.py --dataset flowers --model RN50 --batch_size 128 --num_seeds 5
CUDA_VISIBLE_DEVICES=$device python scripts/run_linear_probe.py --dataset fgvc_aircraft --model RN50 --batch_size 128 --num_seeds 5
CUDA_VISIBLE_DEVICES=$device python scripts/run_linear_probe.py --dataset sun397 --model RN50 --batch_size 128 --num_seeds 5

# for seed in {0..4}
# do
#     python scripts/evaluate_imagenet_captions_.py --model nllb-clip-base --device $device --seed $seed --nonuniform class_weight
#     python scripts/evaluate_imagenet_captions_.py --model nllb-clip-base --device $device --seed $seed --nonuniform variance
#     python scripts/evaluate_imagenet_captions_.py --model nllb-clip-base --device $device --seed $seed --nonuniform class_weight_variance
# done
