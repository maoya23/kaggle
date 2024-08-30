
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

#YAMLファイルを読み込む
with open('../config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

df=pd.read_csv(config['train_path'])

train_df,val_df=train_test_split(df,test_size=0.2,random_state=42)
train_df=train_df.reset_index(drop=True)
val_df=val_df.reset_index(drop=True)

