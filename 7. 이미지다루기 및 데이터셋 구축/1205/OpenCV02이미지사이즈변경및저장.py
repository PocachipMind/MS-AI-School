'''
OpenCV의 resize() 메소드를 이용하여 이미지 크기 변경이 가능하다
'''

import cv2
import matplotlib.pyplot as plt

image_path = './cat.png'

# 이미지 읽기
image = cv2.imread(image_path)

# RGB 타입 변환
image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# 사이즈 변환
image_50x50 = cv2.resize(image, (50,50))

flg, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[1].imshow(image_50x50)
ax[1].set_title("Resize Image")
plt.show()

# 이미지 저장하기
cv2.imwrite("./cat_image_50x50.png", image_50x50)
# 보통 저장할때 png 추천드림
# 함수 사용법 : imwrite( 저장경로 , 저장하고자 하는 이미지)
