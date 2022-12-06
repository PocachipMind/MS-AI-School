'''
순서가 있는 범주형 특성 인코딩
- 순서가 있는 클래스는 순서 개념을 가진 수치값으로 변환해야 합니다.
- 클래스 레이블 문자열을 정수로 매핑하는 딕셔너리를 만들고 이를 필요한 특성에 적용합니다
'''

# 판다스 데이터프레임의 replace()를 사용하여 문자열 레이블을 수치값으로 변환합니다.
import pandas as pd

dataframe = pd.DataFrame(
    {"Score": ["Low", "Low", "Medium", "Medium", "High"]})  # 특성 데이터 생성
scale_mapper = {"Low": 1,
                "Medium": 2,
                "High": 3}  # 매핑 딕셔너리 생성
dataframe["Score"].replace(scale_mapper)  # 특성을 정수로 변환
dataframe = pd.DataFrame(
    {"Score": ["Low", "Low", "Medium", "Medium", "High", "Barely More Than Medium"]})
scale_mapper = {"Low": 1,
                "Medium": 2,
                "Barely More Than Medium": 3,
                "High": 4}  # 매핑 딕셔너리 생성
data = dataframe["Score"].replace(scale_mapper)  # 특성을 정수로 변환
print(data)
