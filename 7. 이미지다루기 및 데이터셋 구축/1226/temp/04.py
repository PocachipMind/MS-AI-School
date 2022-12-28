import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

# 데이터 생성
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70],
                             ])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# TensorDataset 입력으로 사용하고 dataset 지정합니다.
dataset = TensorDataset(x_train, y_train)

# dataloader 
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# model 설계
model = nn.Linear(3, 1) # 한개만 돌릴거면 클래스 굳이 안만들고 이렇게 씀
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

# train loop
epoch_number = 200
for epoch in range(epoch_number + 1):
    for batch_idx , sample in enumerate(dataloader):
        x_train, y_train = sample

        prediction = model(x_train)
        
        # loss
        loss = F.mse_loss(prediction, y_train)

        # loss H(x) 계산
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print("Epoch : {:4d}/{} batch: {}/{} loss : {:.6f}".format(
            epoch, epoch_number, batch_idx+1, len(dataloader), loss.item()
        ))

test_val = torch.FloatTensor([[73, 80, 75]])
pred_y = model(test_val)
print(f"훈련 후 입력이 {test_val} 예측값 : {pred_y}")
# pred_y에 .item()붙이면 숫자로 나옴





