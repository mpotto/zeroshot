#!/bin/bash

device=1

CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_template.py --dataset imagenet1k --model nllb-clip-base --batch_size 128

# for seed in {0..9}
# do
#     python scripts/evaluate_imagenet_captions_.py --model nllb-clip-base --device $device --seed $seed
# done
