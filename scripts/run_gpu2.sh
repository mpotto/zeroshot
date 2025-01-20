#!/bin/bash

device='cuda:2'
model_type='ViT-B-32'

python scripts/evaluate.py --device $device --n_tasks 10 --model_type $model_type
python scripts/evaluate.py --device $device --n_tasks 5 --model_type $model_type