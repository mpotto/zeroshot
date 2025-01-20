import torch
import os
import sys
import open_clip
import json
import argparse
from torch.utils.data import DataLoader

sys.path.extend([".", ".."])
from src.zeroshot import evaluate
from src.data import ImageClassificationDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--template",
    type=str,
    default="imagenet1k",
    help="which prompt template in 'data/prompts.json",
)
parser.add_argument("--device", type=str, default="cuda:0", help="gpu index")
parser.add_argument("--n_tasks", type=int, required=True, help="how many subtasks to split the 250 classes into", choices=[5, 10])
parser.add_argument("--model_type", type=str, required=True, help="which open_clip model to use", choices=['RN50', 'nllb-clip-base', 'ViT-B-32'])
args = parser.parse_args()

DEVICE = args.device
TEMPLATE = args.template
DATA_PATH = "data"
SAVE_PATH = "classifiers"
OUT_PATH = "output"
N_TASKS = args.n_tasks

model_type, pretrained = {
    'RN50': ('RN50', 'yfcc15m'),
    'nllb-clip-base': ('nllb-clip-base', 'v1'),
    'ViT-B-32': ('ViT-B-32', 'datacomp_m_s128m_b4k')
}[args.model_type]

# load data and model
input_filename = os.path.join(DATA_PATH, "eval_df.csv")
class_filename = os.path.join(DATA_PATH, "global_class_df.csv")

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_type, pretrained=pretrained)
model.to(DEVICE)
transforms = preprocess_val

clf_fp = os.path.join(SAVE_PATH, f"{model_type}/{pretrained}/{TEMPLATE}.pt")
if not os.path.exists(clf_fp):
    print(f"No classifier found for '{model_type}' with'{pretrained}' pre-training and '{TEMPLATE}' templates. Exiting.")
else:
    clf = torch.load(clf_fp, map_location='cpu')
    for task_id in range(N_TASKS):
        print(f"\t Running task {task_id + 1}/{N_TASKS}.")

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
        out_fp = os.path.join(OUT_PATH, f"{model_type}/{pretrained}/{TEMPLATE}_{N_TASKS}_tasks_{task_id}.json")
        if os.path.exists(out_fp):
            print(f"Output '{out_fp}' already exists.")
        else:
            print(f"Generating '{out_fp}'.")

            weights = dataset.get_task_weights(clf)
            output = evaluate(model, dataloader, DEVICE, weights)
            os.makedirs(os.path.join(OUT_PATH, f"{model_type}/{pretrained}"), exist_ok=True)
            with open(out_fp, 'w') as file:
                json.dump(output, file)
            print(f"Successfully generated '{out_fp}'.")