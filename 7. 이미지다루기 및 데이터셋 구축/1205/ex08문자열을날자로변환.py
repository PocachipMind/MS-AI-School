'''
# 문자열을 날짜로 변환
• 날짜와 시간을 나타내는 문자열 벡터를 시계열 데이터로 변환
• to_datetime() - format 매개변수에 날짜와 시간 포맷을 지정
• errors 매개변수 - 오류 처리, coerce 옵션값은 문제가 발생해도 에러를 일으키지 않지만 대신 에러가 난 값을 NaT(누락된 값)으로 설정합니다.
'''

import numpy as np
import pandas as pd

data_strings = np.array(['12-05-2022 01:28 PM',
                         '12-06-2022 02:28 PM',
                         '12-07-2022 12:00 AM'])  #문자열

# Timestamp 객체로 변환
for data in data_strings:
    print(pd.to_datetime(data, format='%d-%m-%Y %I:%M %p'))

print()

for data in data_strings:
    print(pd.to_datetime(data, format='%d-%m-%Y %I:%M %p', errors="ignore"))

print()
data = pd.to_datetime(data_strings)
print(data)