from PIL import Image
from matplotlib import pyplot as plt
import cv2

import numpy as np
import time
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

# pip install -U albumentations
import albumentations
from albumentations.pytorch import ToTensorV2

# albumentations Data pipeline
class alb_cat_dataset(Dataset):
    def __init__(self,file_paths, transform = None):
        self.file_paths = file_paths
        self.transform = transform

    def __getitem__(self, index):
        file_path = self.file_paths[index]

        # read an image with opencv
        image = cv2.imread(file_path)

        # convert : BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        start_t = time.time()
        if self.transform is not None:
            image = self.transform(image=image)["image"]
            total_time = (time.time() - start_t)

        return image, total_time

    def __len__(self):
        return len(self.file_paths)

# 기존 torchvision Data pipeline
# 1. dataset class -> image loader -> transform
class CatDataset(Dataset):
    def __init__(self, file_paths, transform = None):
        self.file_paths = file_paths
        self.transform = transform

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        
        # 원래 라면 image label도 반환.
        # Read an image with PIL
        image = Image.open(file_path)

        # transform time check
        start_time = time.time()
        if self.transform : 
            image = self.transform(image)
        end_time = (time.time() - start_time)

        return image, end_time

    def __len__(self):
        return len(self.file_paths)

#### data aug transforms  # compose = 묶어서 하겠다.
# torchvision_transform = transforms.Compose([
#     transforms.Resize((256,256)),
#     transforms.RandomCrop(224), # 랜덤 위치에서 224 만큼 crop
#     transforms.ColorJitter(),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.ToTensor()
# ])
torchvision_transform = transforms.Compose([
    # transforms.Pad(padding=10),
    # transforms.Resize((256, 256)),
    # transforms.CenterCrop(size=(30)),
    # transforms.Grayscale(num_output_channels=1),
    # transforms.ColorJitter(brightness=0.2, contrast=0.3),
    # transforms.GaussianBlur(kernel_size=(3, 9), sigma=(0.1, 5)),
    # transforms.RandomPerspective(distortion_scale=0.7, p=0.5),
    # transforms.RandomRotation(degrees=(0, 100)),
    # transforms.RandomAffine(
    #     degrees=(30, 60), translate=(0.1, 0.3), scale=(0.5, 0.7)),
    # transforms.ElasticTransform(alpha=255.0),
    # transforms.AutoAugment(),
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])
# https://pytorch.org/vision/stable/transforms.html 트랜스폼즈들 있는거 참고용 사이트

# train 보통 트래인용 트랜스폼을 만든다. 
# train과 val을 따로 나눠서 관리한다.

albumentations_transform = albumentations.Compose([
    albumentations.Resize(256, 256),
    albumentations.RandomCrop(224, 224),
    albumentations.HorizontalFlip(),
    albumentations.VerticalFlip(),
    # albumentations.pytorch.transforms.ToTensor()
    ToTensorV2()
])

albumentations_transform_oneof = albumentations.Compose([
    albumentations.Resize(256,256),
    albumentations.RandomCrop(244, 244),
    albumentations.OneOf([
        albumentations.HorizontalFlip(p=1),
        albumentations.RandomRotate90(p=1),
        albumentations.VerticalFlip(p=1)
    ], p =1),
    albumentations.OneOf([
        albumentations.MotionBlur(p=1),
        albumentations.OpticalDistortion(p=1),
        albumentations.GaussNoise(p=1)
    ],p=1),
    ToTensorV2()
])

alb_dataset = alb_cat_dataset(file_paths=['./crazy_cat.png'],
                                transform = albumentations_transform)

cat_dataset = CatDataset(file_paths=['./crazy_cat.png'],
                                transform = torchvision_transform)

alb_oneof_dataset = alb_cat_dataset(file_paths=['./crazy_cat.png'],
                                transform = albumentations_transform_oneof)

total_time = 0
for i in range(100):
    image, end_time = cat_dataset[0]
    total_time += end_time

print("torchvision tiem/image >> ", total_time*10)

alb_total_time = 0
for i in range(100):
    alb_image, end_time = alb_dataset[0]
    alb_total_time += end_time

print("alb time >> ", alb_total_time*10)

oneof_total_time = 0
for i in range(100):
    oneof_image, end_time = alb_oneof_dataset[0]
    oneof_total_time += end_time

print("oneof time >> ", oneof_total_time*10)

# plt.figure(figsize=(10, 10))
# plt.imshow(transforms.ToPILImage()(alb_image).convert("RGB"))
# plt.show()

'''
같은 작업인데 시간차이가 이정도나 남.
torchvision tiem/image >>  13.245420455932617
alb time >>  1.8755578994750977
그래서 보통 alb를 많이씀! 최적화가 잘되있으니까
'''