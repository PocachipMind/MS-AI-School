#  cv2.filter2D()를 활용하여 필터 적용

import cv2
import numpy as np
from utils import image_show

image = cv2.imread('./car.png' )

# Creating out sharpening filter
filter = np.array([[-1,-1,-1] , [-1,9,-1], [-1,-1,-1]])

sharpen_img = cv2.filter2D(image, -1, filter)

# 비교해보기
cv2.imshow("org image", image)
image_show(sharpen_img, "sharpen image")