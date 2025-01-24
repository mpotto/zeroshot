import torch
import os
import sys
import open_clip
import json
import argparse
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

sys.path.extend([".", ".."])
from src.zeroshot import evaluate
from src.data import ImageClassificationDataset

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=3, help="gpu index")
parser.add_argument("--n_tasks", type=int, default=5, help="how many subtasks to split the 250 classes into", choices=[5, 10])
parser.add_argument("--model", type=str, required=True, help="which open_clip model to use", choices=['RN50', 'nllb-clip-base', 'ViT-B-32'])
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

# SEED = 0
# N_TASKS = 5
# DEVICE = 3
# model = 'RN50'

SEED = args.seed
N_TASKS = args.n_tasks
DEVICE = args.device
model = args.model
DATA_PATH = "data"
SAVE_PATH = "classifiers"
OUT_PATH = "output"
SAMPLE_SIZES = [5, 10, 25, 50, 100]

model_type, pretrained = {
    'RN50': ('RN50', 'yfcc15m'),
    'nllb-clip-base': ('nllb-clip-base', 'v1'),
    'ViT-B-32': ('ViT-B-32', 'datacomp_m_s128m_b4k')
}[model]

d = {
    'RN50': 1024,
    'nllb-clip-base': 512,
    'ViT-B-32': 512
}[model]

# load data and model
input_filename = os.path.join(DATA_PATH, "eval_df.csv")
class_filename = os.path.join(DATA_PATH, "global_class_df.csv")

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_type, pretrained=pretrained)
model.to(DEVICE)
transforms = preprocess_val

all_labels = torch.load(os.path.join(DATA_PATH, f"{model_type}_labels.pt"))
all_text_features = torch.load(os.path.join(DATA_PATH, f"{model_type}_text_features.pt"))
clf_fp = os.path.join(DATA_PATH, f"{model_type}_text_features.pt")

def create_global_weights(labels, embeddings, M, seed, num_classes=250):
    torch.manual_seed(seed)
    np.random.seed(seed)
    weights = torch.zeros(size=(250, d))
    for c in range(num_classes):
        sub_embeds = embeddings[labels==c]
        weights[c] = sub_embeds[np.random.choice(len(sub_embeds), min(M, len(sub_embeds)), replace=False)].mean(axis=0)
        weights[c] = F.normalize(weights[c], dim=-1)
    return weights.T


for M in SAMPLE_SIZES:
    clf = create_global_weights(all_labels, all_text_features, M, SEED)
    for task_id in range(N_TASKS):
        print(f"\t Running task {task_id + 1}/{N_TASKS} for sample size {M:03d} and seed {SEED:02d}.")

        dataset = ImageClassificationDataset(input_filename, class_filename, transforms, n_tasks=N_TASKS, task_id=task_id)

        dataloader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            drop_last=False,
        )

        # generate output
        os.makedirs(os.path.join(OUT_PATH, f"{model_type}/imagenet_captions"), exist_ok=True)
        out_fp = os.path.join(OUT_PATH, f"{model_type}imagenet_captions/task_{task_id:02d}_sample_size_{M}_seed_{SEED:02d}.json") 
        if os.path.exists(out_fp):
            print(f"Output '{out_fp}' already exists.")
        else:
            print(f"Generating '{out_fp}'.")

            weights = dataset.get_task_weights(clf)
            output = evaluate(model, dataloader, DEVICE, weights)
            os.makedirs(os.path.join(OUT_PATH, f"{model_type}/{pretrained}"), exist_ok=True)
            with open(out_fp, 'w') as file:
                json.dump(output, file, indent=4)
            print(f"Successfully generated '{out_fp}'.")