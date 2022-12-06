import cv2
import numpy as np
from utils import image_show

image = cv2.imread('./car.png' )

# 커스텀 필터 제작
filter = np.array([[-1, -8, -1 ], [-8 , 38, -8], [-1, -8, -1 ]])

custom_image_filter = cv2.filter2D(image, -1, filter)

image_show(custom_image_filter, "custom_image_filter")
