import numpy as np
#각 열을 기준으로 정렬

array = np.array([[5, 2, 7, 6], [2, 3, 10, 15]])
print("각 열을 기준으로 정렬 전 \n", array)
'''
각 열을 기준으로 정렬 전 
 [[ 5  2  7  6]
 [ 2  3 10 15]]
 '''

array.sort(axis=0)
print("각 열을 기준으로 정렬 후 \n", array)
'''
각 열을 기준으로 정렬 후 
 [[ 2  2  7  6]
 [ 5  3 10 15]]
 '''