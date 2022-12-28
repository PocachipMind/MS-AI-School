# 다중 선형 회귀 클래스로 선언

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 데이터 생성
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70],
                             ])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# class 생성
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear( 3, 1 ) # input 3개, output 1
    
    def forward(self, x):
        return self.linear.forward(x)


# model 정의
model = MultivariateLinearRegressionModel()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

# train
epochs_num = 2000
for epoch in range(epochs_num + 1):

    prediction = model(x_train) # 학습하고자 하는 데이터 전달

    # loss
    loss = F.mse_loss(prediction, y_train)

    # loss 개선
    optimizer.zero_grad()  # 기울기를 0으로 초기화
    loss.backward()        # loss 함수를 미분하여 기울기 계산
    optimizer.step()       # w, b를 업데이트

    if epoch % 100 == 0:
        print("Epoch : {:4d}/{} loss : {:.6f}".format(
            epoch, epochs_num, loss.item()
        ))

new_var = torch.FloatTensor([[73, 82, 72]])
pred_y = model(new_var)
print(f"훈련 후 입력이 {new_var} 예측값 : {pred_y}")