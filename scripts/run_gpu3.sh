#!/bin/bash

device=3

CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_complexity.py --dataset fgvc_aircraft --model ViT-B-32 --batch_size 128
CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_complexity.py --dataset fgvc_aircraft --model nllb-clip-base --batch_size 128
CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_complexity.py --dataset fgvc_aircraft --model RN50 --batch_size 128