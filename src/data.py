import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import logging
import pandas as pd
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split

NUM_CLASSES = 250
# location of directory containing imagenet_images_flickr,
# where files are of the form imagenet_images_flickr/n02110341/2041810051.jpg
# see https://github.com/mlfoundations/imagenet-captions.
DATA_PATH = f"/mnt/ssd/ronak/datasets/imagenet_captions_{NUM_CLASSES}k" 

class ImageClassificationDataset(Dataset):
    def __init__(
            self, 
            input_filename, 
            class_filename,
            transforms,
            n_tasks=25,
            task_id=0,
            img_key="filepath", 
            sep="\t",
        ):
        assert n_tasks in [25, 10, 5, 1]
        assert task_id in list(range(n_tasks))

        logging.debug(f'Loading csv data from {input_filename} for task {task_id + 1}/{n_tasks}.')
        data = pd.read_csv(input_filename, sep=sep)
        classes = pd.read_csv(class_filename, sep=sep)

        self.task_size = NUM_CLASSES // n_tasks
        df, class_df, self.global_to_local = self.classification_transform(data, classes, task_id)
        self.class_names = class_df.sort_values(by="local_label")["class_name"].tolist()

        self.images = df[img_key].map(lambda x: os.path.join(DATA_PATH, x)).tolist()
        self.labels = df["local_label"].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

    def get_task_weights(self, weights):
        task_weights = torch.zeros(size=(len(weights), self.task_size))
        for key in self.global_to_local:
            task_weights[:, self.global_to_local[key]] = weights[:, key]
        return task_weights

    def classification_transform(self, data, classes, task_id):
        # global label has already been randomized
        sub_df = data.loc[(data["global_label"] >= self.task_size * task_id) & (data["global_label"] < self.task_size * (task_id + 1))].copy()
        sub_classes = classes.loc[(classes["global_label"] >= self.task_size * task_id) & (classes["global_label"] < self.task_size * (task_id + 1))].copy()

        global_to_local = {label: i for i, label in enumerate(sub_df["global_label"].unique().tolist())}
        sub_df["local_label"] = sub_df["global_label"].map(lambda x: global_to_local[x])
        sub_classes["local_label"] = sub_classes["global_label"].map(lambda x: global_to_local[x])
        return sub_df, sub_classes, global_to_local

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        labels = self.labels[idx]
        return idx, images, labels
    

class TextClassificationDataset(Dataset):
    def __init__(
            self, 
            input_filename, 
            tokenizer,
            caption_key="title", 
            sep="\t",
        ):

        df = pd.read_csv(input_filename, sep=sep)
        self.captions = df[caption_key].tolist()
        self.labels = df["global_label"].tolist()
        self.tokenize = tokenizer
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        texts = self.tokenize([str(self.captions[idx])])[0]
        labels = self.labels[idx]
        return idx, texts, labels
    
class MultimodalEmbeddingDataset(Dataset):
    def __init__(self, x, y, class_id=None, class_embeds=None):
        self.x = x
        self.y = y
        self.n = len(self.x)
        self.zero_shot = not (class_id is None or class_embeds is None)

        if self.zero_shot:
            self.z = class_id
            self.class_embeds = class_embeds

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.zero_shot:
            return i, self.x[i], self.y[i], self.z[i]
        else:
            return i, self.x[i], self.y[i]

def get_multimodal_dataloaders(
    batch_size, 
    rank, 
    img_embed,
    txt_embed,
    root="/mnt/ssd/ronak/datasets/imagenet_captions_250k",
):
    image_features = np.load(os.path.join(root, f"{img_embed}_image_features.npy"))
    text_features  = np.load(os.path.join(root, f"{txt_embed}_text_features.npy"))
    x_train, x_test, y_train, y_test = train_test_split(image_features, text_features, test_size=0.1, random_state=42)
    val_class_embeds = None
    test_dataset = MultimodalEmbeddingDataset(x_test, y_test)

    train_dataset = MultimodalEmbeddingDataset(x_train, y_train)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size
    )
    print(f"{len(train_dataset):>5,} training samples on rank {rank}.")
    test_dataloader = DataLoader(
        test_dataset, shuffle=True, batch_size=batch_size
    )
    print(f"{len(test_dataset):>5,} validation samples on rank {rank}.")
    return train_dataloader, test_dataloader, val_class_embeds