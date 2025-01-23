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
TEMPLATE = "imagenet_captions"
DATA_PATH = "data"
SAVE_PATH = "classifiers"
NUM_CLASSES = 250

model_type = 'nllb-clip-base'

model_type, pretrained = {
    'RN50': ('RN50', 'yfcc15m'),
    'nllb-clip-base': ('nllb-clip-base', 'v1'),
    'ViT-B-32': ('ViT-B-32', 'datacomp_m_s128m_b4k')
}[model_type]

input_filename = os.path.join(DATA_PATH, "prompt_df.csv")
class_filename = os.path.join(DATA_PATH, "global_class_df.csv")

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

caption_embeddings = torch.zeros(size=(512, NUM_CLASSES)).to(DEVICE)
counts = torch.zeros(size=(1, NUM_CLASSES))

with torch.no_grad():
   for idx, texts, labels in tqdm(dataloader):
      texts = texts.to(DEVICE)
      text_embeddings = model.encode_text(texts)
      text_embeddings = F.normalize(text_embeddings, dim=-1)
      for embedding, label in zip(text_embeddings, labels):
         caption_embeddings[:, label] += embedding
         counts[0, label] += 1


caption_embeddings = caption_embeddings.to('cpu') 
caption_embeddings /= counts
caption_embeddings = F.normalize(caption_embeddings, dim=0) # renormalize

clf_fp = os.path.join(SAVE_PATH, f"{model_type}/{pretrained}/{TEMPLATE}.pt")
torch.save(caption_embeddings, clf_fp)