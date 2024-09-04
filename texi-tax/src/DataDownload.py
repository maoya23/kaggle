
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
kaggle_command = 'kaggle competitions download -c recruit-restaurant-visitor-forecasting -p /data/input/train'
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