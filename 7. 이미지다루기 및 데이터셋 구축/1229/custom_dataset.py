import random

import torch
import os
import glob
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
# 버전 matplotlib >> pip install matplotlib==3.6.1
import numpy as np

# hand data 0~9

class custom_dataset(Dataset):
    def __init__(self, file_path):
        # file_path -> data/train/
        # image 위치 -> data/train/0/*.png
        self.file_path = glob.glob(os.path.join(file_path, "*", "*.png"))
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):

        image_path = self.file_path[index]
        # print(image_path)
        # ./data/train\8\eight_00131.png ( 로컬에서 )

        label = int(image_path.split("\\")[1]) # 로컬에서
        mo = image_path.split("\\")[0].split('/')[2] # 로컬에서

        # label = int(image_path.split("\\")[3]) # VM에서
        # mo = image_path.split("\\")[2] # VM에서

        img = Image.open(image_path).convert('RGB')

        if mo == "train" :
            pass
            if random.uniform(0,1) < 0.2 or img.getbands()[0] == 'L' :
                # Random gray scale from 20%
                img = img.convert('L').convert("RGB")

            if random.uniform(0,1) < 0.2 :
                # Rnadom Gaussian blur from 20%
                gaussianBlur = ImageFilter.GaussianBlur(random.uniform(0.5, 1.2))
                img = img.filter(gaussianBlur)

        else :
            if img.getbands()[0] == 'L' :
                img = img.convert('L').convert('RGB')
        
        img = self.transform(img)

        return img, label

    
    def __len__(self):
        return  len(self.file_path)
        

if __name__ == "__main__":
    train_dataset = custom_dataset("./data/train")

    # print(train_dataset.__len__()) 이런 형식으로도 길이 호출 가능

    _ , ax = plt.subplots(2,4,figsize=(16,10))

    for i in range(8) :
        data = train_dataset.__getitem__(np.random.choice(range(train_dataset.__len__())))

        image = data[0].cpu().detach().numpy().transpose(1,2,0) * 255
        image = image.astype(np.uint32)

        label = data[1]

        ax[i//4][i-(i//4)*4].imshow(image.astype("uint8"))
        ax[i//4][i-(i//4)*4].set_title(label)

    plt.show()