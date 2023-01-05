import argparse
import copy
import os

import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from custom_dataset2 import customDataset
from timm.loss import LabelSmoothingCrossEntropy
from adamp import AdamP
from utils import train, test_species, test_show

def main(opt):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # augmentaion
    train_transfrom = A.Compose([
        # A.Resize(width=224, height=224),
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.8),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.7),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomShadow(p=0.5),
        A.RandomFog(p=0.4),
        A.RandomShadow(p=0.3),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #보통쓰는 정규값 0.485, 0.456, 0.406 / 0.229, 0.224, 0.225  // 귀찮거나 못 외우면 0.5, 0.2
        ToTensorV2()
    ])


    val_transform = A.Compose([
        # A.Resize(width=224, height=224),
        A.SmallestMaxSize(max_size=160),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # dataset
    train_dataset = customDataset(img_path=opt.train_path, transform=train_transfrom)
    val_dataset = customDataset(img_path=opt.val_path, transform=val_transform)
    test_dataset = customDataset(img_path=opt.test_path, transform=val_dataset)

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size= opt.batch_size, shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size= opt.batch_size , shuffle=False)
    test_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)  # 처리 용이하게 batch를 1

    # model call
    net = models.__dict__["resnet50"](pretrained=True)

    # train -> label -> 53  라벨 갯수 맞추기 ctrl 누른 상태에서 models 선택해서 해당 함수 확인해서 fc단 수정
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 53)
    net.to(device)
    # print(net)
    # (fc): Linear(in_features=512, out_features=53, bias=True)

    # loss
    criterion = LabelSmoothingCrossEntropy().to(device)

    # optimizer
    optimizer = AdamP(net.parameters(), lr=opt.lr)

    # scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 90], gamma=0.1)  # 60, 90 에포크 때 lr이 0.1만큼 떨어짐
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamm=0.1)  # 20번 마다 lr 떨어짐

    # model .pt save dir
    save_dir = opt.save_path
    os.makedirs(save_dir, exist_ok=True)
    # (num_epoch, model, train_loader, val_loader, criterion,
    #           optimizer, scheduler, save_dir, device)

    if opt.train_flg == True:
        train(opt.epoch, net, train_loader, val_loader, criterion, optimizer, scheduler, save_dir, device)
    else :
        test_species(test_loader, device)
        # test_show(test_loader, device)

# 사용자에게서 option 값 받기
def parse_opt() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, default="./archive/train", help="train data path")
    parser.add_argument("--val-path", type=str, default="./archive/valid", help="val data path")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--epoch", type=int, default=100, help="epoch number")
    parser.add_argument("--lr", type=float, default=0.001, help="lr number")
    parser.add_argument("--save-path", type=str, default="./weight", help="save path")
    parser.add_argument("--train-flg", type=bool, default=False, help="train or test mode flg")
    parser.add_argument("--test-path", type=str, default='./archive/test', help="test data path")
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
