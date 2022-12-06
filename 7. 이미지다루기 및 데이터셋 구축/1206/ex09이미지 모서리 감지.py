'''
cornerHarris - 해리스 모서리 감지의 OpenCV 구현
해리스 모서리 감지기는 두 개의 경계선이 교차하는 지점을 감지하는 방법으로 사용됩니다.
모서리는 정보가 많은 포인트입니다.
해리스 모서리 감지기는 윈도(이웃, 패치)안의 픽셀이 작은 움직임에도 크게 변하는 윈도를 찾습니다.
cornerHarris 매개변수 block_size : 각 픽셀에서 모서리 감지에 사용되는 이웃 픽셀 크기
cornerHarris 매개변수 aperture : 사용하는 소벨 커널 크기
'''

import cv2
import numpy as np
from utils import image_show

image_path = './edge.png'

# 이미지 읽기
image_read = cv2.imread(image_path)
print(image_read.shape) # (523, 862, 3)

# 모서리 찾기
image_gray = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)
image_gray = np.float32(image_gray)

block_size = 2 # 모서리 감지 매개변수를 설정
aperture = 29
free_parameter = 0.04 # 이웃한 픽셀과 얼마만큼 근접히 할것인지.

detector_response = cv2.cornerHarris( image_gray, block_size, aperture, free_parameter ) # 모서리 감지

# 임계값보다 큰 감지 결과만 남기고 흰색으로 표시합니다.
threshold = 0.02
image_read[detector_response > threshold * detector_response.max()] = [255,255,255]

image_gray = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY) # 흑백으로 변환
image_show(image_gray,"edge(spot)")