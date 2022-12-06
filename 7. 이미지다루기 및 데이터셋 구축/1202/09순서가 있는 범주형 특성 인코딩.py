'''
순서가 있는 범주형 특성 인코딩
- 사이킷런 0.20 이상 부터는 범주형 데이터를 정수로 인코딩하는 OrdinalEncoder 가 추가 되었습니다.
- OrdinalEncoder는 클래스 범주를 순서대로 반환
- 특정 열만 범주형으로 변환하려면 ColumnTransformer와 함께 사용합니다
'''
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

features = np.array([["Low", 10], ["High", 50], ["Medium", 3]])
ordinal_encoding = OrdinalEncoder()
ordinal_encoding.fit_transform(features)
ordinal_encoding_data = ordinal_encoding.categories_

print("ordinal_encoding.categories_", ordinal_encoding_data)
