import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=5,stride=1)
        self.conv2 = nn.Conv2d(in_channels=3,out_channels=10,kernel_size=5,stride=1)
        self.fc1 = nn.Linear(10*12*12, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        print("연산전 x.size >> ", x.size())
        # 연산전 x.size >>  torch.Size([10, 1, 20, 20])
        x = F.relu(self.conv1(x))
        print("conv1 연산 후 x.size >> ",x.size())
        # conv1 연산 후 x.size >>  torch.Size([10, 3, 16, 16])
        x = F.relu(self.conv2(x))
        print("conv2 연산 후 x.size >> ",x.size())
        # conv2 연산 후 x.size >>  torch.Size([10, 10, 12, 12])

        # 차원 감소 후 -> x 값이 어떻게 변하는가?
        x = x.view(-1, 10*12*12)
        print("차원 감소 후 >>", x.size())
        # 차원 감소 후 >> torch.Size([10, 1440])
        x = F.relu(self.fc1(x))
        print("fc1 연산 후 x.size >>", x.size())
        # fc1 연산 후 x.size >> torch.Size([10, 50])
        x = F.relu(self.fc2(x))
        print("fc2 연산 후 x.size >>", x.size())
        # fc2 연산 후 x.size >> torch.Size([10, 10])


cnn = CNN()
output = cnn(torch.randn(10, 1, 20, 20))