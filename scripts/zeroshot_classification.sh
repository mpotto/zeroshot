#!/bin/bash

model='RN50'
pretrained='yfcc15m'
# model="nllb-clip-base"
# pretrained="v1"
# model='ViT-B-32'
# pretrained='datacomp_m_s128m_b4k'
root="/mnt/ssd/ronak/datasets/clip_benchmark"
dataset="cifar10"
templates="cifar10_prompts_full"

clip_benchmark eval --dataset=$dataset \
--task=zeroshot_classification \
--pretrained=$pretrained \
--model=$model \
--output="/home/ronak/zeroshot/output/${model}/${dataset}_${templates}.json" \
--dataset_root=$root \
--custom_template_file="/home/ronak/zeroshot/prompts/${templates}.json" \
--batch_size=64