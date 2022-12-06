'''
이미지를 흐리게 하기 위해서는 각 픽셀을 주변 픽셀의 평균값으로 변환하면 되며, 이렇게 주변 픽셀에
수행되는 연산을 커널(kernel)이라고 한다. 커널이 클수록 이미지가 더 부드러워지게 된다.
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
# from ... import image_show 할수있긴함

def image_show(image, ti="show"):
    cv2.imshow(ti,image)
    cv2.waitKey(0)

image_path = './cat.png'

# 이미지 읽기
image = cv2.imread(image_path)

# 이미지 블러
image_blury = cv2.blur(image, (5,5)) # 보통 33, 55 를 사용한다.
image_show(image_blury)