import numpy as np
# numpy 중복된 원소 제거
array = np.array([1,1,2,2,3,3,4,4,5])
print("중복 처리 전 > ", array) #  중복 처리 전 >  [1 1 2 2 3 3 4 4 5]

print("중복 처리후 > ", np.unique(array)) # 중복 처리후 >  [1 2 3 4 5]

# np.isin() -> 내가 찾는게 있는지 여부 각 index 위치에 true, false
array = np.array([1,2,3,4,5,6,7])

iwantit = np.array([1,2,3,10])

print(np.isin(array, iwantit)) # [ True  True  True False False False False]