# 누락된 클래스 값 대처하기 -2
# 누락된 값을 특성에 가장 자주 등장하는 값으로 채우기
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 범주형 특성을 가진 행렬 생성
x = np.array([[0, 2.10, 1.45],
              [1, 1.18, 1.33],
              [0, 1.22, 1.27],
              [1, -0.21, -1.19]])

# 범주형 특성에 누락된 값이 있는 특성 행렬 만들기
x_with_nan = np.array([[np.nan, 0.87, 1.31], [np.nan, -0.67, -0.22]])
clf = KNeighborsClassifier(3, weights="distance")  # KNN 객체 생성
trained_model = clf.fit(x[:, 1:], x[:, 0])  # 훈련
imputed_val = trained_model.predict(x_with_nan[:, 1:])  # 누락된 값의 클래스를 예측
# 예측된 클래스와 원본 특성을 열로 합칩니다.
x_with_imputed = np.hstack((imputed_val.reshape(-1, 1), x_with_nan[:, 1:]))
data = np.vstack((x_with_imputed, x))  # 두 특성 행렬을 연결

# 누락된 클래스 값 대처하기 -02
# 두 개의 특성 행렬로 합침
# 누락된 값을 특성에서 가장 자주 등장하는 값으로 채우기
x_complete = np.vstack((x_with_nan, x))
imputer = SimpleImputer(strategy="most_frequent")
data_imputer = imputer.fit_transform(x_complete)
print(data_imputer)