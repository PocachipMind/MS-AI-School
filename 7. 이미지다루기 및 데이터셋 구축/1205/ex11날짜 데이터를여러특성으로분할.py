# 날짜와 시간의 열로부터 년, 월, 일, 시, 분에 해당하는 특성을 만들 수 있습니다.
# Series.dt의 시간 속성을 사용합니다.

import pandas as pd

dataframe = pd.DataFrame()
# 다섯 개의 날짜를 만듭니다.
dataframe['date'] = pd.date_range('1/1/2022', periods=150, freq='W')
# 년, 월, 일, 시, 분에 대한 특성을 만듭니다.
dataframe['year'] = dataframe['date'].dt.year
dataframe['month'] = dataframe['date'].dt.month
dataframe['day'] = dataframe['date'].dt.day
dataframe['hour'] = dataframe['date'].dt.hour
dataframe['minute'] = dataframe['date'].dt.minute

print(dataframe.head(5))
