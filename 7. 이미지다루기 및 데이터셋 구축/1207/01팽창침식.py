import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./Billiards.png', cv2.IMREAD_GRAYSCALE)

_, mask = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)

# kernel shape
kernel = []
# rectangle, cross, ellipse로 모양 설정
for i in [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]:
    kernel.append(cv2.getStructuringElement(i, (11, 11)))

titles =['Rectangle', 'Cross', 'Ellipse']
# Kernel = np.ones((3,3), np.uint)
plt.subplot(2,2,1)
plt.imshow(mask, 'gray')
plt.title('origin')

# 침식.   erode를 dilation으로 바꾸면 팽창
for i in range(3):
    erosion = cv2.erode(mask, kernel[i])
    plt.subplot(2,2,i+2)
    plt.imshow(erosion, 'gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()


# 커널이 너무 큰듯하여 3x3으로 다시 시험
kernel = np.ones((3,3),np.uint8)
dilation = cv2.dilate(mask, kernel)
erosion = cv2.erode(mask, kernel)
images = [img, mask, dilation, erosion]
titles = ['origin image', 'mask', 'dilation', 'erotion']

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()