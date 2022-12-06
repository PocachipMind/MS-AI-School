# 대상 픽셀을 강조하는 커널을 정의한 후 filter2D() 메소드를 사용하여 이미지에 적용한다.

import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "./cat.png"

# 이미지 선명하게 ( 커널 생성 : 대상이 있는 픽셀을 강조)
# 컬러 이미지 로드
image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

#RGB 타입으로 변환
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 커널 생성
kernel = np.array([[0 ,-1, 0],
                   [-1, 5,-1],
                   [0 ,-1, 0]])

# 커널 적용
image_sharp = cv2.filter2D(image_rgb, -1, kernel)

fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].imshow(image_rgb)
ax[0].set_title("Original Image")
ax[1].imshow(image_sharp)
ax[1].set_title("sharp Image")
plt.show()
