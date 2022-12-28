# 다중 퍼셉트론으로 손글씨 분류
# 사이킷런에 있는 제공한 이미지 이용

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim

from sklearn.datasets import load_digits

digits = load_digits()
# print(digits)

# 첫번째 샘플을 출력 > .images[인덱스] 8x8
print(digits.images[0])

# 실제 레이블도 숫자 0인지 첫번째 (샘플레이어)샘플의 레이블 확인 > .target[인덱스]
print(digits.target[0])

# 전체 이미지 개수 : 1797
print("전체 이미지 수 : ", len(digits.images))

# 상위 5개만 샘플 이미지 확인
# zip()
"""
image = [1,2,3,4]
label = [사과, 바나나, 자몽, 수박]
이렇게 개수가 동일할때 zip사용가능. 개수가 다르면 짤림.

zip으로 묶음녀 이렇게나올것임
(1, 사과) (2, 바나나)....

"""
image_and_label_list = list(zip(digits.images, digits.target))

for index, (image, label) in enumerate(image_and_label_list[:10]) :
    plt.subplot(2, 5, index +1 )
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('sample : %i'%label)
plt.show()

# 상위 레이블 5개 확인
for i in range(5):
    print(i, "번 index sample label : ", digits.target[i])


# train data 와 label
x = digits.data # 이미지 데이터
y = digits.target # 각 이미지 레이블

model = nn.Sequential(
    nn.Linear(64, 32), # input_layer = 64, hidden_layer_1 = 32
    nn.ReLU(),
    nn.Linear(32, 16), # input_layer = 32, hidden_layer_2 = 16
    nn.ReLU(),
    nn.Linear(16, 10) # input_layer = 16, output_layer = 10
    # CrossEntropyLoss() : output layer = 2 이상인 경우 쓸 수 있지만 우린 안쓴것임.
)
print(model)
"""
Sequential(
  (0): Linear(in_features=64, out_features=32, bias=True)
  (1): ReLU()
  (2): Linear(in_features=32, out_features=16, bias=True)
  (3): ReLU()
  (4): Linear(in_features=16, out_features=10, bias=True)
)
"""
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.int64)

loss_fun = nn.CrossEntropyLoss() # 소프트 맥스 함수를 포함!
optimizer = optim.Adam(model.parameters())

losses = [] # loss 그래프 확인
epoch_number = 100

# 이 구조는 확실히 알아둘 것!
for epoch in range(epoch_number+1):
    output = model(x)
    loss = loss_fun(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print("Epoch : [{:4d}/{}] loss : {:.6f}".format(epoch, epoch_number, loss.item()))
    
    # append
    losses.append(loss.item())

plt.title("loss")
plt.plot(losses)
plt.show()
