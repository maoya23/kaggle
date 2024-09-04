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
/checkpoint
/data
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
import kaggle
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob 
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import shutil
from zipfile import ZipFile
import subprocess
import os
import yaml


with open('../config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)



# ダウンロード先ディレクトリの作成
os.makedirs('/data/input/train', exist_ok=True)

# Kaggleコマンドの実行
kaggle_command = 'kaggle competitions download -c hatena -p /data/input/train'
subprocess.run(kaggle_command, shell=True, check=True)

print('データのダウンロードが完了しました。')



api=KaggleApi()
api.authenticate()

output_path=config['train_path']

api.competition_download_file('recruit-restaurant-visitor-forecasting',
    'air_reserve.csv.zip', path=output_path)

shutil.unpack_archive(config['train_path']+'/'+'recruit-restaurant-visitor-forecasting.zip',config['train_path'])



files = glob.glob(os.path.join(config['train_path'], '*zip'))
from tqdm import tqdm

for item in tqdm(files, desc="ZIPファイルの解凍"):
    with ZipFile(item) as zip_file:
        zip_file.extractall(config['train_path'])

zip_files = glob.glob(os.path.join(config['train_path'], '*.zip'))
for zip_file in tqdm(zip_files, desc="ZIPファイルの削除"):
    os.remove(zip_file)
    print(f"{zip_file}を削除しました。")
'''

with open(competition_name+'/src/DataDownload.py','w') as f:
    f.write(code)

