# 파이토치로 다층 퍼셉트론 구현

import torch
import torch.nn as nn

# GPU 사용가능 여부 파악
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# xor 문제를 풀기 위한 입력과 출력 정의
x = [[0,0], [0,1], [1,0], [1,1]]
y = [[0], [1], [1], [0]]

x = torch.tensor(x, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)

# 참고
# CrossEntropy 경우에는 마지막 레이어 노드수가 2개 이상이여야한다.
# 만약 마지막 층에 1개 output 이라면 BCELoss를 사용할 때가 있다.
# BCELoss 함수를 사용할 경우에는 먼저 마지막 레이어의 값이 0~1로 조정을 해줘야한다.
# 따라서 BCELoss 함수를 쓸 땐 마지막 레이어를 시그모이드 함수를 적용시켜야함 !!

"""
제일먼저 CrossEntropy같은경우에는 마지막 레이어 노드수가 2개 이상이여야 한다.
1개일 경우에는 사용이 안됨.

마지막 레이어가 노드수가 1개일 경우에는 보통 Binary Classification을 할때 사용될수가 있는데
이럴경우 BCELoss를 사용할때가 있다.

BCELoss함수를 사용할 경우에는 먼저 마지막 레이어의 값이 0~1로 조정을 해줘야하기 때문에
단순하게 마지막 레이어를 nn.Linear로 할때 out of range 에러가 뜬다.

따라서 BCELoss함수를 쓸땐 마지막 레이어를 시그모이드함수를 적용시켜줘야 한다.
"""

# model
# 입력층, 은닉층1, 은닉층 2, 은닉층 3, 출력층
model = nn.Sequential(
    nn.Linear(2, 10, bias=True),  # input_layer = 2, hidden_layer 1 -> 10
    nn.Sigmoid(),
    nn.Linear(10, 10, bias=True), # hidden_layer1 = 10, hidden_layer2 = 10
    nn.Sigmoid(),
    nn.Linear(10, 10, bias=True), # hidden_layer2 = 10, hidden_layer3 = 10
    nn.Sigmoid(),
    nn.Linear(10, 1, bias=True),  # hidden_layer3 = 10, output_layer = 1
    nn.Sigmoid()                  # 우리가 사용할 Loss BCELoss 이므로 마지막 레이어를 시그모이드 함수 적용
).to(device)

print(model)
'''
Sequential(
  (0): Linear(in_features=2, out_features=10, bias=True)
  (1): Sigmoid()
  (2): Linear(in_features=10, out_features=10, bias=True)
  (3): Sigmoid()
  (4): Linear(in_features=10, out_features=10, bias=True)
  (5): Sigmoid()
  (6): Linear(in_features=10, out_features=1, bias=True)
  (7): Sigmoid()
)
'''

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 에포크 수행 하겠습니다.
epoch_number = 45000
for epoch in range(epoch_number + 1):
    output = model(x)

    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 100의 배수에 해당되는 에포크 마다 loss print
    if epoch % 100 == 0:
        print(f"Epoch : {epoch} loss : {loss.item()}")


# 인퍼런스 코드
with torch.no_grad() :
    output = model(x)
    predicted = (output >= 0.5).float()
    acc = (predicted == y).float().mean()
    print("모델의 출력값 output\n", output.detach().cpu().numpy())
    print()
    print("모델의 예측값 output\n", predicted.detach().cpu().numpy())
    print()
    print("실제값 (Y) \n", y.cpu().numpy())
    print()
    print("정확도 -> ", acc.item())

    # 실제값 0, 1, 1, 0 줬다면 