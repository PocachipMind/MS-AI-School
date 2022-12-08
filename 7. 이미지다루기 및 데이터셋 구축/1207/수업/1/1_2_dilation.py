import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../Billiards.png', cv2.IMREAD_GRAYSCALE)

_, mask = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)

# kernel shape

kernel = []
for i in [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]:
    kernel.append(cv2.getStructuringElement(i, (11, 11)))

## erosion example
title = ['Rectangle', 'Cross', 'Ellipse']

## dilation example
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.imshow(mask, 'gray')
plt.title('origin')

for i in range(3):
    erosion = cv2.dilate(mask, kernel[i])
    plt.subplot(2, 2, i + 2)
    plt.imshow(erosion, 'gray')
    plt.title(title[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

kernel = np.ones((3, 3), np.uint8)
dilation = cv2.dilate(mask, kernel)
erosion = cv2.erode(mask, kernel)
images = [img, mask, dilation, erosion]
title = ['origin image', 'mask', 'dilation', 'erotion']
plt.figure(figsize=(15, 15))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(title[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
