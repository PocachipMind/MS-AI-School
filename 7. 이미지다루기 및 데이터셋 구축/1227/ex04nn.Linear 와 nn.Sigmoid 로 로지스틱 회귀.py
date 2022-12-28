# 파이토치의 nn.Linear 와 nn.Sigmoid 로 로지스틱 회귀를 구현하기
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# train data -> Tensor
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

# Tensor
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# nn.Sequential()은 Wx+b와 같은 수식과 시그모이드 함수 등과 같은 여러 함수들을 연결해주는 역할을 합니다.
model = nn.Sequential(
    nn.Linear(2, 1),  # input dim = 2 out put dim = 1
    nn.Sigmoid()  # 출력은 시그모이드 함수를 거칩니다.
)

optimizer = optim.SGD(model.parameters(), lr=0.1)
epochs_num = 1000

for epoch in range(epochs_num + 1):

    output = model(x_train)

    # loss
    loss = F.binary_cross_entropy(output, y_train)

    # loss H(x) 계선
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        prediction = output >= torch.FloatTensor([0.5])  # 예측값이 0.5 넘으면 True 간주
        correct_prediction = prediction.float() == y_train  # 실제값과 일치하는 경우만 True
        acc = correct_prediction.sum().item() / len(correct_prediction)  # 정확도 계산
        print("Epoch : {:4d}/{} loss :{:.6f} acc : {:.2f}%".format(
            epoch, epochs_num, loss.item(), acc * 100
        ))


print(model(x_train))
"""
tensor([[0.0283],
        [0.1555],
        [0.2931],
        [0.7868],
        [0.9429],
        [0.9813]], grad_fn=<SigmoidBackward0>)

        y_data = [[0], [0], [0], [1], [1], [1]]
"""
