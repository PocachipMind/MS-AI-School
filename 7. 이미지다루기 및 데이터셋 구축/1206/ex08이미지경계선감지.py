'''
Canny()메소드를 활용하여 경계선을 감지 할 수 있다. Canny()메소드는 그래디언트 임곗값 사이의 저점과고점을 나타내는 두 매개변수를 필요로 하며, 
낮은 임곗값과 높은 임곗값 사이의 가능성 있는 경계선 픽셀은 약한 경계선 픽셀로 간주하고, 
높은 임곗값보다 큰 픽셀은 강한 경계선 픽셀로 간주한다.
'''

import cv2
import numpy as np
from utils import image_show

# 이미지 읽기
image = cv2.imread('./pizza.png')

# 경계선 찾기 ( canny 할 땐 항상 그레이로 해야함 )
image_gray = cv2.imread('./pizza.png', cv2.IMREAD_GRAYSCALE)

# 픽셀 강도의 중간값을 계산
mdeian_intensity = np.median(image_gray) # 이 이미지의 중간값을 찾음
print(mdeian_intensity)

# 중간 픽셀 강도에서 위아래 1표준편차 떨어진 값을 임계값으로 설정
lower_threshold = int(max(0, (1.0 - 0.33) * mdeian_intensity ))
upper_threshold = int(min(255, (1.0 + 0.33) * mdeian_intensity))

# Canny edge Detection 적용
Image_canny = cv2.Canny(image_gray, lower_threshold, upper_threshold)
image_show(Image_canny)