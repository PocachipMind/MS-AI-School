#  머신러닝 분류 알고리즘을 훈련하여 누락된 값을 예측합니다. (k-최근접 이웃KNN) 분류기를 사용
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

x_input = np.array([[0, 2.10, 1.45],
                    [1, 1.18, 1.33],
                    [0, 1.22, 1.27],
                    [1, -0.21, -1.19]])
# 범주형 특성을 가진 행렬 생성
print(x_input)

# 범주형 특성에 누락된 값이 있는 특성 행렬을 만듭니다.
x_with_nan = np.array([[np.nan, 0.87, 1.31], [np.nan, -0.67, -0.22]])
clf = KNeighborsClassifier(3, weights="distance")  # KNN 객체 생성
trained_model = clf.fit(x_input[:, 1:], x_input[:, 0])  # 훈련
imputed_val = trained_model.predict(x_with_nan[:, 1:])  # 누락된 값의 클래스를 예측
# 예측된 클래스와 원본 특성을 열로 합칩니다.
x_with_imputed = np.hstack((imputed_val.reshape(-1, 1), x_with_nan[:, 1:]))
data = np.vstack((x_with_imputed, x_input))  # 두 특성 행렬을 연결
print(data)
