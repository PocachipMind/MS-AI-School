# 감지 된 모서리에 흰색 원 처리하기
import cv2
import numpy as np
from utils import image_show

image_path = './edge.png'
image_read = cv2.imread(image_path)
image_gray = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)

# 감지할 모서리 개수
corners_to_detect = 4
minimum_quality_score = 0.05
mininum_distance = 25

# 모서리 감지
corners = cv2.goodFeaturesToTrack(image_gray, corners_to_detect,minimum_quality_score,mininum_distance)

for corner in corners:
    x, y = corner[0]
    cv2.circle(image_read, (int(x), int(y)), 10, (0,255,0), -1) # 모서리마다 회색 원을 그림(뒷배경이 하얀색이므로)

image_gray_temp =cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY) # 흑백 이미지로 변환

image_show(image_gray_temp,"check edge with circle")