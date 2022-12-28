# 파이토치로 리니어 만들어봄 ( 선형 회귀 )

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 랜덤 시드 설정
torch.manual_seed(1)

# 실습을 위한 기본셋팅 훈련데이터 x_train , y_train 을 선언
x_train = torch.FloatTensor(([1], [2], [3]))
y_train = torch.FloatTensor(([2], [4], [6]))

# x_train 와 shape 출력
# print(x_train, x_train.shape)  # shape or size
# print(y_train, y_train.shape)  # shape or size
"""
tensor([[1.],
        [2.],
        [3.]]) torch.Size([3, 1])
tensor([[2.],
        [4.],
        [6.]]) torch.Size([3, 1])
"""

# 가중치와 편향의 초기화 직선 -> w and b
# requires_grad=True -> 학습을 통해 계속 값이 변경되는 변수
w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 가설 세우기
# 직선의 방정식
hypothesis = x_train * w + b

# loss fn 선언 하기
# 평균 제곱 오차 선언
loss = torch.mean((hypothesis - y_train) ** 2)
# print(loss)

# 경사하강법 구현 하기
optimizer = optim.SGD([w, b], lr=0.01)

# 기울기 0 으로 초기화
optimizer.zero_grad()
loss.backward()

# 학습 진행
epoch_num = 2000

# epoch : 전체 훈련 데이터가 학습에 한번 사용된 주기 ( 1 epoch > 한번 전체 다 훑엇다 )
# train loop
for epoch in range(epoch_num+1):

    # 1. 가설 -> model
    hypothesis = x_train * w + b

    # loss
    loss = torch.mean((hypothesis - y_train) ** 2)

    # loss H(x) 개선
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() # 옵티마이저 갱신

    # 100 번 마다
    if epoch % 100 == 0:
        print("Epoch {:4d}/{} W : {:.3f} b : {:.3f} loss : {:.6f}".format(
            epoch, epoch_num, w.item(), b.item(), loss.item()
        ))


# 러닝 레이트가 너무 높으면 loss가 0으로 나올수있음
"""
Epoch    0/2000 W : 0.187 b : 0.080 loss : 18.666666
Epoch  100/2000 W : 1.746 b : 0.578 loss : 0.048171
Epoch  200/2000 W : 1.800 b : 0.454 loss : 0.029767
Epoch  300/2000 W : 1.843 b : 0.357 loss : 0.018394
Epoch  400/2000 W : 1.876 b : 0.281 loss : 0.011366
Epoch  500/2000 W : 1.903 b : 0.221 loss : 0.007024
Epoch  600/2000 W : 1.924 b : 0.174 loss : 0.004340
Epoch  700/2000 W : 1.940 b : 0.136 loss : 0.002682
Epoch  800/2000 W : 1.953 b : 0.107 loss : 0.001657
Epoch  900/2000 W : 1.963 b : 0.084 loss : 0.001024
Epoch 1000/2000 W : 1.971 b : 0.066 loss : 0.000633
Epoch 1100/2000 W : 1.977 b : 0.052 loss : 0.000391
Epoch 1200/2000 W : 1.982 b : 0.041 loss : 0.000242
Epoch 1300/2000 W : 1.986 b : 0.032 loss : 0.000149
Epoch 1400/2000 W : 1.989 b : 0.025 loss : 0.000092
Epoch 1500/2000 W : 1.991 b : 0.020 loss : 0.000057
Epoch 1600/2000 W : 1.993 b : 0.016 loss : 0.000035
Epoch 1700/2000 W : 1.995 b : 0.012 loss : 0.000022
Epoch 1800/2000 W : 1.996 b : 0.010 loss : 0.000013
Epoch 1900/2000 W : 1.997 b : 0.008 loss : 0.000008
Epoch 2000/2000 W : 1.997 b : 0.006 loss : 0.000005
"""
