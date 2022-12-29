import pandas as pd

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader


class customDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        self.x1 = df.iloc[:, 0].values
        self.x2 = df.iloc[:, 1].values
        self.x3 = df.iloc[:, 2].values
        self.y = df.iloc[:, 3].values
        self.length = len(df)

    def __getitem__(self, index):
        x = torch.FloatTensor([self.x1[index], self.x2[index], self.x3[index]])
        y = torch.FloatTensor([self.y[index]])

        return x, y

    def __len__(self):
        return self.length


class customModel(nn.Module):
    def __init__(self):
        super(customModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer(x)
        return x


train_dataset = customDataset("./dataset02.csv")
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = customModel().to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10001):
    losses = 0.0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss

    losses = losses / len(train_loader)

    if (epoch+1) % 100 == 0:
        print(f"Epoch : {epoch+1:4d}, loss : {losses:.4f} ")

# 더 이상 학습하지 않겠다.
with torch.no_grad():
    model.eval()
    inputs = torch.FloatTensor(
        [[89, 92, 75], [75, 64, 50], [38, 58, 63], [33, 42, 39], [23, 15, 32]]
    ).to(device)
    output = model(inputs)

    print("-------------------------")
    print(output)
    print(output >= torch.FloatTensor([0.5]).to(device))
