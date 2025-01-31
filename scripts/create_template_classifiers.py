import torch
import json
import open_clip
import pandas as pd
import os
import sys

sys.path.extend(["..", "."])
from src.zeroshot import zero_shot_classifier

MODELS = [
    ('RN50', 'yfcc15m'),
    ('nllb-clip-base', 'v1'),
    ('ViT-B-32', 'datacomp_m_s128m_b4k')
]
DEVICE = "cuda:0"
TEMPLATE = "imagenet1k"
DATA_PATH = "data/"
OUT_PATH = "classifiers/"

# get class names and prompt templates
df = pd.read_csv(os.path.join(DATA_PATH, f"global_class_df.csv"), header=0, sep="\t")
df = df.sort_values(by="global_label") # global label indexes classifier
classnames = df["class_name"].tolist()

with open(os.path.join(DATA_PATH, 'prompts.json'), 'r') as file:
    prompt_dict = json.load(file)
templates = prompt_dict[TEMPLATE]

for model_type, pretrained in MODELS:
    fp = os.path.join(OUT_PATH, f"{model_type}/{pretrained}/{TEMPLATE}.pt")
    if os.path.exists(fp):
        print(f"Zero-shot classifier for '{model_type}' with'{pretrained}' pre-training and '{TEMPLATE}' templates already exists.")
    else:
        print(f"Creating zero-shot classifier for '{model_type}' with'{pretrained}' pre-training and '{TEMPLATE}' templates.")

        # load model
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_type, pretrained=pretrained)
        model.to(DEVICE)
        tokenizer = open_clip.get_tokenizer(model_type)

        classifier = zero_shot_classifier(model, tokenizer, classnames, templates, DEVICE)
        print(f"Successfully created zero-shot classifier for '{model_type}' with'{pretrained}' pre-training and '{TEMPLATE}' templates.")
        os.makedirs(os.path.join(OUT_PATH, f"{model_type}/{pretrained}"), exist_ok=True)
        torch.save(classifier, fp)
    print()