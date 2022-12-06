import numpy as np
# numpy 배열 객체 복사
array1 = np.arange(0,10)
array2 = array1
array2[0] = 99
print("그냥 대입 결과 (array1) : ",array1) # array1[0]이 같이 변경됨
print()
'''
그냥 대입 결과 (array1) :  [99  1  2  3  4  5  6  7  8  9]
'''

# 내부적으로 array1과 array2 동일한 주소를 씀 그래서 array2를 수정해도 array1도 변경됨
# 이러한 문제를 해결하는 방법으로
# 복사하는 함수를 사용함

array1 = np.arange(0,10)
array2 = array1.copy()
array2[0] = 99
print("함수 사용 대입 (array1) : ",array1)
print("함수 사용 대입 (array2) : ",array2) # array2만 변경됨
'''
함수 사용 대입 (array1) :  [0 1 2 3 4 5 6 7 8 9]
함수 사용 대입 (array2) :  [99  1  2  3  4  5  6  7  8  9]
'''