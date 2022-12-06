import cv2
from utils import image_show

image = cv2.imread('./cat.png')

# image 10x10 픽셀 크기로 변환
image_color_10x10 = cv2.resize(image, (10,10))
image_shape_info = image_color_10x10.flatten().shape
image_color_10x10.flatten() # flatten이란 다차원 배열을 1차원 공간으로 변경
image_show(image_color_10x10 , "image_color_10x10") 

# image 225x255 픽셀 크기로 변환
image_color_225x255 = cv2.resize(image, (225, 255))
image_color_225x255.flatten()
image_show(image_color_225x255 , "image_color_225x255") 

print(image_color_225x255.shape) 
# (255, 225, 3)
print(image_color_225x255.flatten().shape) # 이미지 데이터를 1차원 벡터로 변환하고 차원을 출력
# (172125,)