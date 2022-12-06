# 순서가 없는 범주형 데이터 처리
# 사이킷런의 LabelBinarize를 사용하여 문자열 타깃 데이터 원 - 핫 인코딩 진행
import numpy as np
from sklearn.preprocessing import OneHotEncoder

feature = np.array([["Texas", 1], ["California", 1],
                   ["Texas", 3], ["Delaware", 1], ["Texas", 1]]) # 여러개의 열이 있는 특성 배열 생성
print(feature)
one_hot_encoder = OneHotEncoder(sparse=False) # 희소배열을 반환이 기본값이며 sparse = False로 지정하면 밀집 배열 반환
# OneHotEncoder -> 입력 특성 배열을 모두 범주형으로 인식하여 변환합니다.
one_hot_encoder.fit_transform(feature)
one_hot_encoder_data = one_hot_encoder.categories_  # categories_ 속성으로 클래스 확인
