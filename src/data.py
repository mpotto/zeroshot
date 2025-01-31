import torch
from torch.utils.data import Dataset
import logging
import pandas as pd
from PIL import Image
import numpy as np
import os

NUM_CLASSES = 250
# location of directory containing imagenet_images_flickr,
# where files are of the form imagenet_images_flickr/n02110341/2041810051.jpg
# see https://github.com/mlfoundations/imagenet-captions.
DATA_PATH = "~" 

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