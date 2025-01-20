import torch
import os
import sys
import open_clip
import json
from torch.utils.data import DataLoader

sys.path.extend([".", ".."])
from src.zeroshot import evaluate
from src.data import ImageClassificationDataset

DEVICE = "cuda:0"
TEMPLATE = "imagenet1k"
DATA_PATH = "data"
SAVE_PATH = "classifiers"
OUT_PATH = "output"

# TODO: Make this work for all tasks and ids.
task_id = 4
n_tasks = 10
model_type = 'ViT-B-32'
pretrained = 'laion2b_s34b_b79k'

# load data and model
input_filename = os.path.join(DATA_PATH, "eval_df.csv")
class_filename = os.path.join(DATA_PATH, "global_class_df.csv")


model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_type, pretrained=pretrained)
model.to(DEVICE)
transforms = preprocess_val

dataset = ImageClassificationDataset(input_filename, class_filename, transforms, n_tasks=n_tasks, task_id=task_id)

dataloader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
    drop_last=False,
)

# generate output
clf_fp = os.path.join(SAVE_PATH, f"{model_type}/{pretrained}/{TEMPLATE}.pt")
if not os.path.exists(clf_fp):
    print(f"No classifier found for '{model_type}' with'{pretrained}' pre-training and '{TEMPLATE}' templates already exists.")
else:
    out_fp = os.path.join(OUT_PATH, f"{model_type}/{pretrained}/{TEMPLATE}_{n_tasks}_tasks_{task_id}.json")
    if os.path.exists(out_fp):
        print(f"Output '{out_fp}' already exists.")
    else:
        print(f"Generating '{out_fp}'.")
        clf = torch.load(clf_fp, map_location='cpu')

        weights = dataset.get_task_weights(clf)

        output = evaluate(model, dataloader, DEVICE, weights)
        print(f"Successfully generated '{out_fp}'.")
        os.makedirs(os.path.join(OUT_PATH, f"{model_type}/{pretrained}"), exist_ok=True)
        json.save(output, out_fp)