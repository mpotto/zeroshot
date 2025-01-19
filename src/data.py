from torch.utils.data import Dataset
import logging
import pandas as pd
from PIL import Image
import numpy as np
import os

NUM_CLASSES = 250
DATA_PATH = f'/mnt/ssd/ronak/datasets/imagenet_captions_{NUM_CLASSES}k'

def classification_transform(data, n_tasks, task_id, classes):
    # global label has already been randomized
    task_size = NUM_CLASSES // n_tasks

    sub_df = data.loc[(data["global_label"] >= task_size * task_id) & (data["global_label"] < task_size * (task_id + 1))].copy()
    sub_classes = classes.loc[(classes["global_label"] >= task_size * task_id) & (classes["global_label"] < task_size * (task_id + 1))].copy()

    global_to_local = {label: i for i, label in enumerate(sub_df["global_label"].unique().tolist())}
    sub_df["local_label"] = sub_df["global_label"].map(lambda x: global_to_local[x])
    sub_classes["local_label"] = sub_classes["global_label"].map(lambda x: global_to_local[x])
    return sub_df, sub_classes


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
        assert n_tasks in [25, 10, 5]
        assert task_id in list(range(n_tasks))
        logging.debug(f'Loading csv data from {input_filename} for task {task_id + 1}/{n_tasks}.')
        data = pd.read_csv(input_filename, sep=sep)
        classes = pd.read_csv(class_filename, sep=sep)
        df, class_df = classification_transform(data, n_tasks, task_id, classes)
        self.class_names = class_df.sort_values(by="local_label")["class_name"].tolist()

        self.images = df[img_key].map(lambda x: os.path.join(DATA_PATH, x)).tolist()
        self.labels = df["local_label"].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        labels = self.labels[idx]
        return idx, images, labels