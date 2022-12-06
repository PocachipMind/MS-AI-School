'''
# 시간대 데이터 처리
• 시계열 데이터에서 시간대 정보를 추가하거나 변환할 수 있습니다.
• 판다스 객체에는 시간대가 없습니다
• 판다스는 시간대 객체를 만들때 tz 매개변수를 사용하여 시간대를 추가할 수 있습니다.
• 판다스의 Series 객체는 모든 원소에 tz_localize와 tz_convert를 적용합니다.
'''

import pandas as pd

data = pd.Timestamp('2022-12-05 01:04:00') # dataitem을 만듦

data_in_london = data.tz_localize(tz='Europe/London') # 시간대 지정
print(data_in_london) # datetime 확인

data_in_london.tz_convert('Africa/Abidjan') # 시간대 변환

# 세개의 날짜를 만듭니다
dates = pd.Series(pd.date_range('2/2/2002', periods=3, freq='M'))
temp = dates.dt.tz_localize('Africa/Abidjan') # 시간대 지정

print(temp)