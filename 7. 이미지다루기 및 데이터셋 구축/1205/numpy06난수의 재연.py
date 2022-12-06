import numpy as np
# 난수의 재연 ( 실행마다 결과 동일 )
# 랜덤한 값이 실행할때 마다 변경됨
# 변경안되도록 고정하는 방법 seed
# 학습 결과가 변경되지 않도록 하는 방법

np.random.seed(7777)
print(np.random.randint(0, 10, (2,3)))
'''
[[6 0 9]
 [8 3 8]]
'''