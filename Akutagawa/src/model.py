
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
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import os
from tqdm import tqdm

#YAMLファイルを読み込む
with open('../config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

    
    # Pytorch Lightningを使用するためのClass
class EarlyStopper:
    def __init__(self, verbose=True, path=config['checkpoint_path'], patience=3):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.__early_stop = False
        self.val_loss_max = np.Inf
        self.path = path

        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path),exist_ok=True)
        
    @property
    def early_stop(self):
        return self.__early_stop
        
        
    def update(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, val_loss)
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.__early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model, val_loss)
            self.counter = 0

    def save_checkpoint(self, model, val_loss):
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_loss_max:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_max = val_loss
    
        
    def load_checkpoint(self, model):
        if self.verbose:
            print(f'Loading model from last checkpoint with validation accuracy {self.val_loss_max:.6f}')
        model.load_state_dict(torch.load(self.path))
        return model

'''
class Trainer(nn.Module):
    def __init__(self,model_name,train_dataloader,val_dataloader,loss_function,optimizer,sheduler,epoch,device):
        self.model_name = model_name
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.sheduler = sheduler
        self.epoch = epoch
        self.device = device

    def train(self):
        model.train()
        for epoch in range(self.epoch):
            total_loss = 0
            progress_bar = tqdm(self.train_dataloader, desc=f'Epoch {epoch + 1}/{self.epoch}', leave=False)
            for batch in progress_bar:
                input_ids = batch['ids'].to(self.device)
                attention_mask = batch['mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                loss = self.loss_function(outputs, labels)
                total_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()
                
                progress_bar.set_postfix({'Loss': total_loss / len(self.train_dataloader)})
    
        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['ids'].to(self.device)
                attention_mask = batch['mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                loss = self.loss_function(outputs, labels)
                val_loss += loss.item()
        
            # Update the learning rate scheduler
        self.scheduler.step()

        early_stopping.update(val_loss/len(self.val_dataloader), model)

        if early_stopping.early_stop:
            print('Early stopping !')
            break
        
        print(f'Epoch {epoch + 1}/{config["EPOCH"]}: Train Loss = {total_loss / len(self.train_dataloader)}, Val Loss = {val_loss / len(self.val_dataloader)}')

        model = early_stopping.load_checkpoint(model)

'''