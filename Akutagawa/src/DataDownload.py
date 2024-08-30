
import os
import urllib.request
import zipfile
import requests


url="https://s3.ap-northeast-1.amazonaws.com/nishika.assets.private/competitions/1/data/data.zip?response-content-disposition=attachment%3B%20filename%3Ddata.zip&AWSAccessKeyId=ASIA3NMWWMCV5MMS6LBR&Signature=JkEO0hYZ0XmaQ3efd53ziCfLumI%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEBIaDmFwLW5vcnRoZWFzdC0xIkgwRgIhAOzwmVHWWGCl5ol1MIzMoYyG7DuxXXlDq2hIm9Ls%2F7AgAiEAtQr4FmHqklytp%2Bgwg5Hu5zz44X5njOh9Epi8pbz1nU0qiwQI%2B%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgw3ODQ2ODQ0NDE3NzEiDCxs7PZE835injzJGirfA6dKEBghaNH4DC%2FzN%2BUPy%2BAnoZk5SHWGGItPkHGqUI8jRQu2icwyseagRCiPg8E6YQ%2FwYrcsYe02CgN05JH3uqXreYF6nM7zDO0xgvOb1R5vPVC%2BH4k4CP6IyyrB3DAA5j0Br2hi1I698moFKxgkB711QOqC55V4Wtilh3eZCm8uDbc96Kl4V%2FXA1v5W2s9eHg%2BW%2FGmLLKnpz9oBVMrg7LEYQ6ZJ7IAmKOGk%2B%2BLaO0wMB0AXE9DbycN3XAJ0rR3x1j7QC1Ixrq92uq5oq72BYjF5Z%2BJ75iO%2F2NadTIQH70FS3O8w%2BpZqQqNRQRZibX%2F7EjDP13JlEwlYJVHv1fOTe3L72nEKlyCG2I7GGVByMOJaS8e828%2BVUX01sktCqiRDsT2DSFtcMYQ5xpuXSbvG1%2FxSmVQWM96LZV55af07vmaZ7TmO8GQWnNFCZfTPr8OvDZ0qucx70leDVaMvZRAF4stBsNt1eLLJ6VYeuncIuJy9EbATsdKSfmoyZz4NAxgPg5GP%2B%2FGerdGQqVrERsu6G5D45n8nARC6iMmWd%2FoguEo8NoEiE3jcpL7I5odbbHQmMstZTExh1UZUX%2BCJWQl5%2FnIUxL80Fq9ycZkDr1TD76eUCven%2BwYevjNr5UHvSzTiMOOoy7UGOqQBzuQMsWaFxNdZ7fk81ECVxkmpEi2TXBbPiNv4Vw8OftEy4ibP10ecRGb8GJazMvXGYt6K6l7PO2fqq08lHC1NaYcDfyQSi0tJ5J0WIiGbwcK84OrRxnyj8gZEzKJBUzG6k80w7fGuqtI0VBTsc6bmHYwBQn7K5JPjC1M9NNIj8ijvvSJQsQSGjBoaMoIotIwSPSczc3EvOWSHGBiBiWa4MeSVRUU%3D&Expires=1723016508"
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
