'''
GaussianBlur() 함수의 세 번째 매개변수는 X축(너비) 방향의 표준편차이며, 0으로 지정하면 ((너비-1)0.5-
1)0.3+0.8과 같이 계산된다.
'''
import cv2
from utils import image_show
import numpy as np

# 이미지 경로
image_path = "./cat.png"

# 이미지 읽기 처리
image = cv2.imread(image_path)

# 가우시안 블러
image_very_blurry = cv2.GaussianBlur(image,(5,5),0)
image_show(image_very_blurry)