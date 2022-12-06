#시작과 마지막 날짜를 사용해 불리언 조건을 만들어 날짜 벡터에서 하나 이상의 원소를 선택할 수 있습니다.
# 날짜 열을 데이터프레임의 인덱스로 지정하고 loc를 사용해 슬라이싱할 수 있습니다.

import pandas as pd

dataframe = pd.DataFrame()
# datetime 생성
#periods 매개변수는 date_range함수에 전달된 시작날짜와 종료날짜를 periods 매개변수에 전달된 기간만큼
#동일하게 나누어 출력해줍니다.
dataframe['date'] = pd.date_range('1/1/2022', periods=100000, freq='H')

# 두 datetime 사이의 샘플을 선택합니다.
dataframe[(dataframe['date'] > '2022-1-1 01:00:00') & (dataframe['date'] <= '2022-1-1 04:00:00')]
dataframe = dataframe.set_index(dataframe['date']) # datetime 만듭니다.

temp = dataframe.loc['2002-1-1 01:00:00':'2022-1-1 04:00:00']
print(temp)


'''
2022-01-01 00:00:00 2022-01-01 00:00:00
2022-01-01 01:00:00 2022-01-01 01:00:00
2022-01-01 02:00:00 2022-01-01 02:00:00
2022-01-01 03:00:00 2022-01-01 03:00:00
2022-01-01 04:00:00 2022-01-01 04:00:00
'''