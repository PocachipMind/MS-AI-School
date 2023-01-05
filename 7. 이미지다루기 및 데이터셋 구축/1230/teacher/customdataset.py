import glob
import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
# from albumentations.pytorch import ToTensorV2
# import albumentations as A
# from torchvision.transforms import transforms
# pip install albumentations==1.2.1


class customDataset(Dataset) :
    def __init__(self, img_path, transform=None):
        # dataset / train / * /*.jpg
        self.all_img_path = glob.glob(os.path.join(img_path,"*","*.jpg"))
        self.class_names = os.listdir(img_path)
        self.class_names.sort()
        self.transform = transform
        self.all_img_path.sort()
        self.labels = []

        for path in self.all_img_path:
            self.labels.append(self.class_names.index(path.split('\\')[1]))
        self.labels = np.array(self.labels)

    def __getitem__(self, item):
        image_path = self.all_img_path[item]
        image = cv2.imread(image_path)
        label = self.labels[item]
        label = int(label)
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label, image_path

    def __len__(self):
        return len(self.all_img_path)
