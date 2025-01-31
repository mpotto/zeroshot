import torch
import os
import sys
import open_clip
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.extend(["..", "."])
from src.data import TextClassificationDataset


DEVICE = "cuda:3"
DATA_PATH = "data"
SAVE_PATH = "classifiers"
NUM_CLASSES = 250

model_type = sys.argv[1]

model_type, pretrained = {
    'RN50': ('RN50', 'yfcc15m'),
    'nllb-clip-base': ('nllb-clip-base', 'v1'),
    'ViT-B-32': ('ViT-B-32', 'datacomp_m_s128m_b4k')
}[model_type]

d = {
    'RN50': 1024,
    'nllb-clip-base': 512,
    'ViT-B-32': 512
}[model_type]

input_filename = os.path.join(DATA_PATH, "prompt_df.csv")
# class_filename = os.path.join(DATA_PATH, "global_class_df.csv")

# load model
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_type, pretrained=pretrained)
model.to(DEVICE)
tokenizer = open_clip.get_tokenizer(model_type)

dataset = TextClassificationDataset(input_filename, tokenizer)

dataloader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
    drop_last=False,
)

all_labels, all_text_features, all_idx = [], [], []

with torch.no_grad():
   for idx, texts, labels in tqdm(dataloader):
      texts = texts.to(DEVICE)
      text_embeddings = model.encode_text(texts)
      text_embeddings = F.normalize(text_embeddings, dim=-1)
      all_text_features.append(text_embeddings)
      all_idx.append(idx)
      all_labels.append(labels)



all_labels = torch.cat(all_labels).cpu().detach()
all_text_features = torch.cat(all_text_features).cpu().detach()
all_idx = torch.cat(all_idx).cpu().detach()

# Because the indices were random, we reorder them to be in line with the original dataset order.
reorder = torch.argsort(all_idx)
torch.save(all_labels[reorder], os.path.join(DATA_PATH, f"{model_type}_labels.pt"))
torch.save(all_text_features[reorder], os.path.join(DATA_PATH, f"{model_type}_text_features.pt"))