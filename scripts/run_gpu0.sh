#!/bin/bash


# CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_template.py --dataset $dataset --model ViT-B-32 --batch_size 128
# CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_template.py --dataset $dataset --model nllb-clip-base --batch_size 128
# CUDA_VISIBLE_DEVICES=$device python scripts/run_zeroshot_template.py --dataset $dataset --model RN50 --batch_size 128

python scripts/generate_prompts.py --dataset=flowers --num_outputs=100
python scripts/generate_prompts.py --dataset=dtd --num_outputs=100
python scripts/generate_prompts.py --dataset=fgvc_aircraft --num_outputs=100