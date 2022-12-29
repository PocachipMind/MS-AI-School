import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from custom_dataset import custom_dataset
from torchvision import models

from timm.loss import LabelSmoothingCrossEntropy
# pip install timm < 많은 모델이 있음.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# dataset
train_data = custom_dataset("./data/train") # VM에선 "D:\\data\\train\\" 넣어줌
test_data  = custom_dataset("./data/test")

# dataloader
train_loader = DataLoader(train_data,batch_size=12, shuffle=True)
test_loader = DataLoader(test_data, batch_size=12, shuffle=False)

# model call
net = models.__dict__["resnet18"](pretrained=True)
net.fc = nn.Linear(512,10)
net.to(device)


# loss function
criterion = LabelSmoothingCrossEntropy()
# this is better than nn.CrossEntropyLoss 라고 합니다.
criterion = criterion.to(device)

# optimizer
optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)

net.train()
total_step = len(train_loader)
curr_lr = 0.001
best_score = 0

num_epochs = 100
for epoch in range(num_epochs + 1) :
    total_loss = 0
    for i,(images, labels) in enumerate(train_loader) :
        images = images.to(device)
        labels = labels.to(device)
        output = net(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i+1) % 100 == 0 :
            print("{} / {}".format(12*(i+1), train_data.__len__()))

    net.eval()
    score = 0
    total = 0
    for i,(images, labels) in enumerate(test_loader) :
        images = images.to(device)
        labels = labels.to(device)
        output = net(images)

        total += images.size(0)
        _, argmax = torch.max(output, 1)
        score += (labels == argmax).sum().item()
    print("Epoch : {}, Loss : {:.4f}".format(
        epoch+1, total_loss / total_step
    ))

    avg = (score / total * 100)
    print("Accuracy : {:.2f}\n".format(avg))
    net.train()

    if best_score < avg :
        best_score = avg
        torch.save(net.state_dict(), "./best.pt")