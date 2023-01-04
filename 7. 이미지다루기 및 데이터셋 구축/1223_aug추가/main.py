import os

from dataset_temp import custom_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torch
import hy_parameter
from torchvision import models
import torch.nn as nn
from utils import train, validate, save_model


# device
# 윈도우 기반 그래픽 카드 엔비디아 사용하고 계신경우
# device = torch.device("cuda" if torch.cuda.is_available else "cpu")
# m1 m2 칩셋 사용하시는분
# device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

device = torch.device("cpu")

# train aug
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
train_loader = DataLoader(train_dataset, batch_size=hy_parameter.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=hy_parameter.batch_size, shuffle=False)

# model call
net = models.__dict__["resnet18"](pretrained=True)

# pretrained = true num_classes 4 수정방법
net.fc = nn.Linear(512,4)
net.to(device)

# criterion
criterion = nn.CrossEntropyLoss()

# optimizer
optim = torch.optim.Adam(net.parameters(), lr=hy_parameter.lr)

# model save dir
model_save_dir= "./model_save"
os.makedirs(model_save_dir, exist_ok=True)

train(number_epoch=hy_parameter.epoch, train_loader=train_loader, val_loader=val_loader,
      criterion=criterion, optimizer=optim, model=net, save_dir=model_save_dir, device=device)