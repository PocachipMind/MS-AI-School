import numpy as np
# Numpy 가장 많이 사용되는 함수
# 1. 원소 정렬

# def -> 오름차순 정렬 형태
array = np.array([15, 20, 5, 12, 7])
np.save("./array.npy", array)
array_data = np.load('./array.npy')
print(array_data) # [15 20  5 12  7]
array_data.sort()
print("오름차순 >>", array_data) # 오름차순 >> [ 5  7 12 15 20]

# 내림차순 정렬
print("내림 차순 >>", array_data[::-1]) # 내림 차순 >> [20 15 12  7  5]

# numpy.flip() 을 사용하여 뒤집기 가능!