#!/bin/bash

device='cuda:1'
model_type='nllb-clip-base'

python scripts/evaluate.py --device $device --n_tasks 10 --model_type $model_type
python scripts/evaluate.py --device $device --n_tasks 5 --model_type $model_type