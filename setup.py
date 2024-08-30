import os
from pathlib import Path
import yaml
import subprocess

#コンペの名前をここで設定。これによってコンペごとのdirsができる
competition_name='texi-tax'

#dirsの作成
os.makedirs(competition_name,exist_ok=True)
os.makedirs(competition_name+'/config',exist_ok=True)
os.makedirs(competition_name+'/data',exist_ok=True)
os.makedirs(competition_name+'/data/input',exist_ok=True)
os.makedirs(competition_name+'/data/output',exist_ok=True)
os.makedirs(competition_name+'/checkpoint',exist_ok=True)
os.makedirs(competition_name+'/src',exist_ok=True)

#gitignoreのファイルを作っておく
ignore_path=competition_name+'/.gitignore'
file_path_obj = Path(ignore_path)
if not file_path_obj.exists():
    file_path_obj.touch(exist_ok=True)

code_ignore='''
*
!/config
!/src
!README.md
!.gitignore
'''

with open(competition_name+'/.gitignore','w') as f:
    f.write(code_ignore)



#yamlファイルを作成せするためのファイルをここでつくる
config_path=competition_name+'/config/config.yaml'

#config.yamlに書く内容をここで決定する。これはデフォルトにして使い回す
file_path_obj = Path(config_path)
if not file_path_obj.exists():
    file_path_obj.touch(exist_ok=True)


data={
    'batchsize':256,
    'lr':0.001,
    'epoch':50,

    'train_path':'../data/input/train',
    'test_path':'../data/input/test',
    'output_path':'../data/output',
    'checkpoint_path':'../checkpoint',
    'device':'cuda'
    }

with open(config_path, 'w', encoding='utf-8') as f:
    yaml.dump(data, f, default_flow_style=False,allow_unicode=True)



#pythonの実行ファイルを作成する。
list=['model','preprocess','DataDownload','train','test']

for item in list:
    model_path=competition_name+'/src/'+item+'.py'
    file_path_obj = Path(model_path)
    if not file_path_obj.exists():
        file_path_obj.touch(exist_ok=True)

#train,testのipynbも一応作っておく
for item in 'train','test':
    model_path=competition_name+'/src/'+item+'.ipynb'
    file_path_obj = Path(model_path)
    if not file_path_obj.exists():
        file_path_obj.touch(exist_ok=True)


code='''
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
'''

#model,train,testのファイルにcodeを書き込む
for item in 'train','test','model':
    with open(competition_name+'/src/'+item+'.py','w') as f:
        f.write(code)


#DataDownloadの方にも書き込んでおく
code='''
import os
import urllib.request
import zipfile
import requests


url="https://download.pytorch.org/tutorial/hymenoptera_data.zip"
urlData = requests.get(url).content
file_path='../data/input/train.zip'

with open(file_path, 'wb') as f:
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        for chunk in response.iter_content(1024):
            f.write(chunk)
    else:
        print(f"Download failed. Status code: {response.status_code}")

if os.path.exists(file_path):
  with zipfile.ZipFile(file_path) as existing_zip:
    existing_zip.extractall('../data/input')
else:
  print(f"The file {file_path} does not exist.")

os.remove('../data/input/train.zip')
'''

with open(competition_name+'/src/DataDownload.py','w') as f:
    f.write(code)

