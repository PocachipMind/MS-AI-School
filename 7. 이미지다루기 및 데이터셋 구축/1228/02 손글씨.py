from utils import *
from PIL import Image, ImageOps
import os

train_image_path = "./dataset/train_image/"
train_data = image_file(train_image_path)
print(len(train_data))
# train data 갯수 : 568

test_image_path = "./dataset/test_image/"
test_data = image_file(test_image_path)
print(len(test_data))
# test data 갯수 : 318

train_image_resize = False
if train_image_resize == True:
    for i in train_data:
        f_name = i.split('/')
        # ['.', 'dataset', 'train_image', '9\\nine_00157.jpg']
        f_name = f_name[-1].split('\\')[0]
        # os.path.basename 하면 파일이름 나옴
        
        img = Image.open(i)
        img = ImageOps.exif_transpose(img)
        img_new = expand2square(img, (0,0,0)).resize((400,400))
        # 저장
        file_name = os.path.basename(i)
        file_name = file_name.split('.')
        file_name = file_name[0]
        os.makedirs(f"./data/train/{f_name}/", exist_ok=True)
        img_new.save(f"./data/train/{f_name}/{file_name}.png")

test_image_resize = False
if test_image_resize == True:
    for i in test_data:
        f_name = i.split('/')
        # ['.', 'dataset', 'test_image', '9\\nine_00157.jpg']
        f_name = f_name[-1].split('\\')[0]
        # os.path.basename 하면 파일이름 나옴
        
        img = Image.open(i)
        img = ImageOps.exif_transpose(img)
        img_new = expand2square(img, (0,0,0)).resize((400,400))
        # 저장
        file_name = os.path.basename(i)
        file_name = file_name.split('.')
        file_name = file_name[0]
        os.makedirs(f"./data/test/{f_name}/", exist_ok=True)
        img_new.save(f"./data/test/{f_name}/{file_name}.png")

