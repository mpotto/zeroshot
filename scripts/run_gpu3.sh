#!/bin/bash

device=3

python scripts/create_ideal_embeddings.py RN50
python scripts/create_ideal_embeddings.py ViT-B-32
python scripts/create_ideal_embeddings.py nllb-clip-base