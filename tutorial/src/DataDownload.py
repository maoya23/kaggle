
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
