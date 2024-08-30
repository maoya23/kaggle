import glob
import random
import pickle
import yaml
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import os

#YAMLファイルを読み込む
with open('../config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def calculate_weight(df):
    n_label=len(df['author'])

    label_count=df['author'].value_counts(sort=False)
    label_count_dict=label_count.to_dict()

    label_weight=[]
    for i in range(n_label):
        rate=(label_count_dict[i]/len(df))*100
        weight=100-rate
        label_weight.append(weight)

    return label_weight


class MyDataset(Dataset):
    def __init__(self,data,tokenizer,label_weight):
        self.data=data
        self.tokenizer=tokenizer
        self.class_weight=label_weight
        self.sample_weights=[0]*len(data)
        self.max_length=256

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_row=self.data.iloc[index]
        text=data_row['body']
        labels=data_row['author']

        encoding=self.tokenizer.encoding_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            retuen_tensors='pt'
        )


        return dict(
            input_ids=encoding['input_ids'].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            token_type_ids=encoding["token_type_ids"].flatten(),
            labels=torch.tensor(labels)
        ) 

    def get_sampler(self): #--4
        for idx, row in self.data.iterrows():
            label = row['author']
            class_weight = self.class_weights[label]
            self.sample_weights[idx] = class_weight
        sampler = WeightedRandomSampler(self.sample_weights, num_samples=len(self.sample_weights), replacement=True)
        return sampler           



class DataModule(pl.LightningDataModule):
        def __init__(self, train_df, val_df, test_df, tokenizer,label_weight):
            super().__init__()
            self.train_df = train_df
            self.val_df = val_df
            self.test_df = test_df
            self.tokenizer = tokenizer
            self.batch_size = config['batchsize']
            self.label_weight=label_weight

        def setup(self): #--2
            self.train_dataset = MyDataset(self.train_df, self.tokenizer,self.label_weight)
            self.valid_dataset = MyDataset(self.val_df, self.tokenizer,self.label_weight)
            self.test_dataset = MyDataset(self.test_df, self.tokenizer,self.label_weight)
            self.train_sampler = self.train_dataset.get_sampler()
            self.valid_sampler = self.valid_dataset.get_sampler()

        def train_dataloader(self): #--3
            return DataLoader(self.train_dataset, batch_size=self.batch_size["train"], num_workers=os.cpu_count(), sampler=self.train_sampler)

        def val_dataloader(self): #--4
            return DataLoader(self.valid_dataset, batch_size=self.batch_size["val"], num_workers=os.cpu_count(), sampler=self.valid_sampler)

        def test_dataloader(self): #--5
            return DataLoader(self.test_dataset, batch_size=self.batch_size["test"], num_workers=os.cpu_count())            