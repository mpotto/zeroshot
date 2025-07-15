#!/bin/bash

device=0

# CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_template.py --dataset imagenet1k --model ViT-B-32 --batch_size 128
# CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_complexity.py --dataset sun397 --model RN50 --batch_size 128
# CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_retrieval.py --dataset flickr8k --model nllb-clip-base --batch_size 128 --num_seeds 5
# CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_retrieval.py --dataset mscoco_captions --model nllb-clip-base --batch_size 128 --num_seeds 5

# python train_variance_reduction.py --dataset imagenet_captions_250k --experiment_name clip_gpt --device cuda:0 --seed=0

# CUDA_VISIBLE_DEVICES=$device python scripts/run_linear_probe.py --dataset dtd --model nllb-clip-base --batch_size 128 --num_seeds 5
# CUDA_VISIBLE_DEVICES=$device python scripts/run_linear_probe.py --dataset flowers --model nllb-clip-base --batch_size 128 --num_seeds 5
# CUDA_VISIBLE_DEVICES=$device python scripts/run_linear_probe.py --dataset fgvc_aircraft --model nllb-clip-base --batch_size 128 --num_seeds 5
# CUDA_VISIBLE_DEVICES=$device python scripts/run_linear_probe.py --dataset sun397 --model nllb-clip-base --batch_size 128 --num_seeds 5

# for seed in {0..4}
# do
#     python scripts/evaluate_imagenet_captions_.py --model RN50 --device $device --seed $seed --nonuniform class_weight
#     python scripts/evaluate_imagenet_captions_.py --model RN50 --device $device --seed $seed --nonuniform variance
#     python scripts/evaluate_imagenet_captions_.py --model RN50 --device $device --seed $seed --nonuniform class_weight_variance
# done

