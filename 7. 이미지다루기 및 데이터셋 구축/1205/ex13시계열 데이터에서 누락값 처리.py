# 시계열 데이터에서는 보간(interpolation)방법을 사용하여 누락된 값으로 생긴 간격을 채웁니다.
# 보간은 누락된 값의 양쪽 경계를 잇는 직선이나 곡선을 사용하여 적절한 값을 에측함으로써 비어 있는 간극을 채우는 기법입니다.
# 시간 간격이 일정하고, 데이터가 노이즈로 인한 변동이 심하지 않고 누락된 값으로 인한 빈 간극이 작을 때 보간방법이 유용합니다.
# 두 포인트 사이의 직선이 비선형이라고 가정하면 interpolate의 method 매개변수를 사용해 다른 보간 방법을 지정할 수 있습니다.
# 누락된 값의 간격이 커서 전체를 간격을 보간하는 것이 좋지 않을 때는 limit 매개변수를 사용하여 보간 값의 개수를 제한하고
# limit_direction 매개변수에서 마지막 데이터로 앞쪽 방향으로 보간할지 그 반대로 할지 지정할 수 있습니다.
# 누락된 값을 이전에 등장한 마지막 값으로 대체할 수 있다.
# 누락된 값을 그 이후에 등장한 최초의 값으로 대체할 수 있다

import pandas as pd
import numpy as np

time_index = pd.date_range('01/01/2022', periods=5, freq='M')
dataframe = pd.DataFrame(index=time_index) # 데이터 프레임을 만들고 인덱스 지정
print(dataframe)

dataframe['Sales'] = [1.0, 2.0, np.nan, np.nan, 5.0] # 누락된 값이 있는 특성 생성
data = dataframe.interpolate() # 누락된 값을 보간
data01 = dataframe.ffill() # 앞쪽으로 Forward-fill
data02 = dataframe.bfill() # 뒤쪽으로 Back-fill
data03 = dataframe.interpolate(method="quadratic") # 비선형의 경우 보간 방법 변경
# 보간 방향 지정
data04 = dataframe.interpolate(limit=1,limit_direction='forward')
