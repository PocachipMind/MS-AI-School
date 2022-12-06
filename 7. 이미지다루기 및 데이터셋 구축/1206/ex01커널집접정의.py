'''
커널을 직접 정의한 후 filter2D() 메소드를 통해 이미지에 적용하는 것도 가능하다.
생성된 커널을 이미지에 적용 시 중앙 원소가 변환되는 픽셀이며, 나머지는 그 픽셀의 이웃이 된다.
'''
import cv2
from utils import image_show
import numpy as np

# 이미지 경로
image_path = "./cat.png"

# 이미지 읽기 처리
image = cv2.imread(image_path)

# 커널 생성 처리
kernel = np.ones((10,10)) / 25.0 # 모두 더하면 1이 되도록 정규화
image_kernel = cv2.filter2D(image, -1, kernel)
print(kernel)
image_show(image_kernel)