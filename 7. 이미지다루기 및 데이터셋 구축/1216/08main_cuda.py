import torch.nn as nn
from collections import defaultdict
import torch
from tqdm import tqdm
import cv2
import os
import copy
import glob
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import numpy as np

from torchvision import models
from albumentations.pytorch import ToTensorV2



"""
RuntimeError: Input type (torch.cuda.ByteTensor) and weight type (torch.cuda.FloatTensor) should be the same
 >>> 입력이 0~255로 표현되는 Byte(uint8)형태이고, 반면 모델의 가중치는 Float32이기 때문에 발생하는 에러입니다.
 >>> 해결방법 : customdataset 에서 image = image.float() 형태로 변경하기 

Target size (torch.Size([64])) must be the same as input size (torch.Size([64, 1]))
 >>> 타겟 차원수가 맞지 않아서 발생하는 문제 아마 이진분류 loss 함수를 사용하다보니 생기는 문제
 >>> 해결방법 : train, val 함수에서  targets = targets.unsqueeze(1) 차원 추가

RuntimeError: result type Float can't be cast to the desired output type Long
 >>> BCEWithLogitsLoss로 다중 레이블 이진 분류를 수행하지만 RunTimeError "RuntimeError: 결과 유형 Float를 원하는 출력 유형 Long으로 캐스트할 수 없습니다"가 발생합니다.
 >>> 해결방법 : loss 계산하는 부분에 targets.float() 변경하기 
 >>> 모델 가중치가 Float32 인데 들어오는건 Long 타입이라문제가 발생한거니 Float형태로 변경
"""

# 엔비디아 GPU 사용하시는분은 요고 주석 풀고 device 사용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class catvsdogDataset(Dataset):
    def __init__(self, image_file_path, transform=None):
        self.image_file_paths = glob.glob(
            os.path.join(image_file_path, "*", "*.jpg"))
        self.transform = transform

    def __getitem__(self, index):
        # image loader
        image_path = self.image_file_paths[index]
        image = cv2.imread(image_path)
        # converter BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # label
        label_temp = image_path.split("\\")
        label_temp = label_temp[1]

        if "cat" == label_temp:
            label = 0
        elif "dog" == label_temp:
            label = 1

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        image = image.float()
        return image, label

    def __len__(self):
        return len(self.image_file_paths)


# aug
train_transform = A.Compose([
    A.Resize(height=224, width=224),
    # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(height=224, width=224),
    ToTensorV2()
])

# dataset
train_dataset = catvsdogDataset("./dataset/train/", transform=train_transform)
val_dataset = catvsdogDataset("./dataset/val/", transform=val_transform)


def visualize_augmentation(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([
        t for t in dataset.transform if not isinstance(t, (ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


def calculate_accuracy(output, target):
    output = torch.sigmoid(output) >= 0.5
    target = target == 1.0
    return torch.true_divide((target == output).sum(dim=0), output.size(0)).item()


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name} : {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


params = {
    "model": "resnet18",
    "lr": 0.001,
    "batch_size": 64,
    "num_workers": 2,
    "epoch": 10
}

# model loader
model = models.__dict__[params["model"]](pretrained=True)
model.fc = nn.Linear(512, 1)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

train_loader = DataLoader(train_dataset, batch_size=params["batch_size"],
                          shuffle=True, )

val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False,
                        )


def save_model(model, save_dir, file_name='last.pt'):
    # save model
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, file_name)
    if isinstance(model, nn.DataParallel):
        print("multi GPU activate")
        torch.save(model.module.state_dict(), output_path)
    else:
        print("single GPU activate")
        torch.save(model.state_dict(), output_path)


def train(train_loader, model, criterion, optimizer, epoch, device):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for i, (image, target) in enumerate(train_loader):
        images = image.to(device)
        targets = target.to(device)
        targets = targets.unsqueeze(1)

        output = model(images)
        loss = criterion(output, targets.float())

        accuracy = calculate_accuracy(output, targets)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch : {epoch}. Train.     {metric_monitor}".format(
                epoch=epoch, metric_monitor=metric_monitor)
        )


def validate(val_loader, model, criterion, epoch, device):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (image, target) in enumerate(val_loader):
            images = image.to(device)
            targets = target.to(device)
            targets = targets.unsqueeze(1)

            output = model(images)
            loss = criterion(output, targets.float())
            accuracy = calculate_accuracy(output, targets)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)

            stream.set_description(
                "Epoch : {epoch}. val.     {metric_monitor}".format(
                    epoch=epoch, metric_monitor=metric_monitor)
            )


save_dir = "./weights"

for epoch in range(1, params["epoch"] + 1):
    train(train_loader, model, criterion, optimizer,
          epoch, device)
    validate(val_loader, model, criterion, epoch, device)

save_model(model, save_dir)
