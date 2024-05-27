import sklearn
from sklearn.preprocessing import *
import numpy as np
from numpy import *

# pip install sklearn > 더이상 쓰이지 않으므로 pip install scikit-learn 으로 변경해서 입력.

# 정규화
# 전체구간을 0~100으로 설정하여 데이터를 관찰하는방법, 특정데이터의위치를확인할 수 있게해줍니다.
# 0 ~ 1 사이의 범위로 데이터를 표준화
def normalization(data):
    data_mm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    return data_mm

# 표준화
# 평균까지의 거리로, 2개 이상의 대상이 단위가 다를때, 대상 데이터를 같은 기준으로 볼 수 있게 해준다.
# 모집단이 정규분포를 따르는 경우에 N(0,1)N(0,1) 인 표준정규분포로 표준화 하는 작업
def numpy_standardization(data):
    """
    (각데이터 - 평균(각열)) / 표준편차(각열)
    """
    std_data = (data - np.mean(data, axis=0) / np.std(data, axis=0))
    return std_data


def main():
    data = np.random.randint(30, size=(6, 5))
    # print(data)
    std_data_temp = numpy_standardization(data)
    print(std_data_temp)

    no_data = normalization(data)
    print(no_data)


if __name__ == '__main__':
    main()
