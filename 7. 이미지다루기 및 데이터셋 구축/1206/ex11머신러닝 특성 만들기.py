'''
• 이미지를 머신러닝에 필요한 샘플로 변환하려면 넘파이의 flatten()을 사용합니다.
• Flatten()은 이미지 데이터가 담긴 다차원 배열을 샘플값이 담긴 벡터로 변환
• 이미지가 흑백일 때 각 픽셀은 하나의 값으로 표현됩니다.
• 컬럼 이미지라면 각 픽셀이 하나의 값이 아니라 여러 개의 값으로 표현됩니다.
• 이미지의 모든 픽셀이 특성이 되기 때문에 이미지가 커질수록 특성의 개수도 크게 늘어납니다
'''

import cv2
from utils import image_show

image_gray = cv2.imread('./cat.png', cv2.IMREAD_GRAYSCALE)

image_10x10 = cv2.resize(image_gray, (10, 10)) # 이미지를 10x10 픽셀 크기로 변환
image_10x10.flatten() # 이미지 데이터를 1차원 벡터로 변환

image_show(image_10x10, "image_10x10")