# 날짜간의 차이 계산
# 판다스의 TimeDelta 데이터 타입을 사용하면 두 지점 사이의 시간 변화를 기록한 특성을 계산합니다

import pandas as pd

dataframe = pd.DataFrame()

# 두 datetime 특성을 만듭니다.
dataframe['Arrived'] = [pd.Timestamp('01-01-2022'), pd.Timestamp('01-04-2022')]
dataframe['Left'] = [pd.Timestamp('01-01-2022'), pd.Timestamp('01-06-2022')]

print(dataframe['Arrived'],dataframe['Left'])
print()

# 특성 사이의 차이를 계산
print(dataframe['Left'] - dataframe['Arrived'])
print()

# 특성 간의 기간 계산
data = pd.Series(delta.days for delta in (dataframe['Left'] - dataframe['Arrived']))

print(data)
