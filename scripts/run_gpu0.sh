#!/bin/bash

device='cuda:0'
model_type='RN50'

python scripts/evaluate.py --device $device --n_tasks 10 --model_type $model_type
python scripts/evaluate.py --device $device --n_tasks 5 --model_type $model_type