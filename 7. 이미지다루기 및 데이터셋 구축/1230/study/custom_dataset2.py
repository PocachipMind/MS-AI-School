import glob
import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random
# from albumentations.pytorch import ToTensorV2
# import albumentations as A
"""
# 시드 고정
random_seed =7777
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
"""

class customDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.all_img_path = glob.glob(os.path.join(img_path, "*", "*.jpg"))
        # print(self.all_img_path)
        self.transform = transform
        self.class_names = os.listdir(img_path)
        self.class_names.sort()
        self.all_img_path.sort()
        self.labels = []
        for path in self.all_img_path:
            self.labels.append(self.class_names.index(path.split('\\')[1]))
        self.labels = np.array(self.labels)

    def __getitem__(self, item):
        image_path = self.all_img_path[item]
        image = cv2.imread(image_path)
        # print(image)
        label = self.labels[item]
        label = int(label)
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label
        # return image, label, image_path


        # print(image_path, label)

    def __len__(self):
        return len(self.all_img_path)

train_dataset = customDataset('./archive/train')
#C:\AI SCHOOL\1230\archive\train\ace of clubs

# for i in train_dataset:
#     print()

