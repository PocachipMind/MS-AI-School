# 같은 크기의 이미지 블렌딩 실험

import cv2
import matplotlib.pyplot as plt
import numpy as np

large_img = cv2.imread('./ex_image.png')
watermark = cv2.imread('./ex_image_logo.png')

print('large_image size >>', large_img.shape)
print('watermarker_image size >>', watermark.shape)

img1 = cv2.resize(large_img, (800,600))
img2 = cv2.resize(watermark, (800,600))

# 혼합 진행

# 처음 0.5 값 설정 
blended = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

# 높은 값으로 전달 
# blended = cv2.addWeighted(img1, 0.8, img2, 0.2, 0)

# 1로 설정 
# blended = cv2.addWeighted(img1, 1, img2, 1, 0)

cv2.imshow("img show", blended)
cv2.waitKey(0)