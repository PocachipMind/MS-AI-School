import cv2
import matplotlib.pyplot as plt
import numpy as np

# image loading and input image -> gray
img_gray = cv2.imread('./Billiards.png', cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY_INV)

# 3x3 커널 생성
kernel = np.ones((3,3), np.uint8) 

dilation = cv2.dilate(mask, kernel)
'''
[[1 1 1]
 [1 1 1]
 [1 1 1]]
'''
titles = ['image', 'mask', 'dilation']
images = [img_gray, mask, dilation]

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()

'''
확장된 이미지를 만들고 이를 위해 함수를 사용합니다 cv2.dilate() 함수에 대한 입력은 마스크 이미지와 커널입니다. 
여기서 우리는 커널을 픽셀 크기가 3x3 작은 정사각형으로 정의 할 수 있습니다.
이 이미지의 행렬의 유형은 부호 없는 정수여야 합니다. (unit8)
'''