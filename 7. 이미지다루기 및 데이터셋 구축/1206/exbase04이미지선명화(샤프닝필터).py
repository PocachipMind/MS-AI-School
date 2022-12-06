# 샤프닝 필터 버전 – 멕시칸 햇 또는 라플라시안 필터
# 이미지의 가장자리를 보존함.

import cv2
import numpy as np
from utils import image_show

image = cv2.imread('./car.png' )

# Creating maxican hot filter
# 5x5 필터
filter = np.array([[0,0,-1,0,0] , [0,-1,-2,-1,0], [-1,-2,16,-2,-1], [0,-1,-2,-1,0], [0,0,-1,0,0]])

# 3x3 필터
# filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# Applying cv2.filter2D function on our Cybertruck image
mexican_hot_image = cv2.filter2D(image, -1, filter)

image_show(mexican_hot_image, "mexican_hot_image")
# 저장해서 비교 가능
# cv2.imwrite('./mexican_hat_5x5.png', mexican_hat_image)