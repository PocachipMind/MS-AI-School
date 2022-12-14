import torch
import numpy as np

data = [[1,2],[3,4]]

# 토치용 데이터로 만들기
x_data = torch.tensor(data)
print(x_data)
'''tensor([[1, 2],
        [3, 4]])'''
print(x_data.shape) # torch.Size([2, 2])

# numpy 배열로부터 생성
# ap_array = np.array(data).reshape
# x_np = torch.from_numpy(ap_array)
# print(x_np)

x_ones = torch.ones_like(x_data)
print(f'Ones Tensor >> \n', x_ones )

x_rand = torch.rand_like(x_data, dtype = torch.float)
print(x_rand)


for_shape = (2,4)
randn_tensor = torch.rand(for_shape)
ones_tensor = torch.ones(for_shape)
zeros_tensor = torch.zeros(for_shape)

print(f'Random Tensor >> \n {randn_tensor} \n')
print(f'Ones Tensor >> \n {ones_tensor} \n')
print(f'Zeros Tensor >> \n {zeros_tensor} \n')


tensor = torch.rand(3,4)

print(f'shape of tensor : {tensor.shape}')
print(f'data type of tensor : {tensor.dtype}')
print(f'device tensor is stored on : {tensor.device}')

tensor = torch.rand(3, 4)
if torch.cuda.is_available() :
        tensor = tensor.to('cuda')
else:
        tensor = tensor.to('cpu')
print()
print(f'device tensor is stored on : {tensor.device}')

tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)
'''
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
'''

# 텐서 합치기 torch.cat 많이 사용함
tensor = torch.ones(4, 4)
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

print(tensor*tensor) # 요소 별 곱하기
# tensor.mul(tensor) 도 가능

# 행렬 곱
print(tensor)
print(tensor.matmul(tensor.T))
print(tensor @ tensor.T)

# 바꿔치기
tensor.add_(5)
print(tensor)
'''
바꿔치기 연산은 메모리를 일부 절약하지만,기록(history)이
즉시 삭제되어 도함수(derivative)계산에문제가 발생할 수 있습니다.
따라서,사용을 권장하지 않습니다.
'''

### 텐서 numpy 벼열 형태로 변환
t = torch.ones(5)
print('tensor -> ',t)
# [1, 1, 1, 1, 1]
n = t.numpy()
print('numpy -> ', n)

### CPU상의 텐서와 Numpy 배열은 메모리 공간을 공유함. 하나를 변경하면 다른 하나도 변경됨.
t.add_(1)
print('numpy -> ', n)

# numpy에서 torch로
n = np.ones(5)
t = torch.from_numpy(n)

# 이 또한 메모리 공유함.
print(n)
print(t)
np.add(n, 1, out=n)
print(n)
print(t)

# 뷰(View) - 원소의 수를 유지하면서 텐서의 크기 변경. 매우 중요!
t = np.array([[[0,1,2],[3,4,5]], [[6, 7, 8], [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft)

print(ft.view([-1,3])) # ft tensor (?, 3) size changed
print(ft.view([-1,3]).shape)

print(ft.view([-1,1,3]))
print(ft.view([-1,1,3]).shape)


# squeeze : 1인 차원을 제거한다.
# 임의의 3X1
ft = torch.FloatTensor([[0],[1],[2]])
print(ft)
print(ft.shape)

print(ft.squeeze())
print(ft.squeeze().shape)

# unsqueeze : 특정 위치에 1인 차원을 추가한다.
# View나 unsqueeze를 통해 차원을 추가할 수 있다.

# unsqueeze
ft = torch.FloatTensor([0, 1, 2])
print(ft)
print(ft.shape)


print(ft.unsqueeze(0)) # 인덱스가 0부터 시작하므로 0은 첫번째 차원을 의미한다.
print(ft.unsqueeze(0).shape)

# view
print(ft.view(1, -1))
print(ft.view(1, -1).shape)