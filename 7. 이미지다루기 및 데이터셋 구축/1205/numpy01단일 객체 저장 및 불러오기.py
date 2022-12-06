import numpy as np

# 단일 객체 저장 및 불러오기
array = np.arange(0, 10)
print(array) # [0 1 2 3 4 5 6 7 8 9]


# .npy 파일에다가 저장하기
np.save('./save.npy', array)

# 불러오기
result = np.load('./save.npy')

print("result >>",result) # result >> [0 1 2 3 4 5 6 7 8 9]
