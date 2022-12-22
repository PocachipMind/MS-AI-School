from dataset_temp import custom_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torch
import hy_parameter
from torchvision import models
import torch.nn as nn
from utils import train, validate, save_model
import os

# device
# 윈도우 기반 그래픽 카드 엔비디아 사용하고 계신경우
# device = torch.device("cuda" if torch.cuda.is_available else "cpu")
# m1 m2 칩셋 사용하시는분( 맥 )
# device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

# cpu로 설정해봄 
device = torch.device("cpu")

# train aug : Cimpose와 ToTensorV2 는 친구임.
train_transform = A.Compose([
    A.Resize(height=224, width=224),
    ToTensorV2()
])
# val aug
val_transform = A.Compose([
    A.Resize(height=224, width=224),
    ToTensorV2()
])

# dataset
train_dataset = custom_dataset("./data/train", transform=train_transform)
val_dataset = custom_dataset("./data/val", transform=val_transform)

# dataloader
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)  # batch_size = 데이터가 64개보다 적을 수 있으므로
val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False)

# model call
net = models.__dict__['resnet18'](pretrained=True) # 프리트레인 되도록이면 하는게 좋음.

# pretrained = true num_classes 4 수정 방법
net.fc = nn.Linear(512, 4) # 학습 갯수가 4개라서 4로 설정 변경한것.
net.to(device)

# criterion
criterion = nn.CrossEntropyLoss()

# optimizer
optim = torch.optim.Adam(net.parameters(), lr=hy_parameter.lr)

# model save dir
model_save_dir= "./model_save"
os.makedirs(model_save_dir, exist_ok=True)

# def train(number_epoch, train_loader, val_loader, criterion, optimizer, model, save_dir, device)
train(number_epoch=hy_parameter.epoch, train_loader=train_loader, val_loader=val_loader,
      criterion=criterion, optimizer=optim, model=net, save_dir=model_save_dir, device=device)