import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../Billiards.png', cv2.IMREAD_GRAYSCALE)

_, mask = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)

title = ['image', 'mask', 'dilation', 'erosion', 'opening', 'closing']

kernel = np.ones((3,3), np.uint8)
dilation = cv2.dilate(mask, kernel)
erosion = cv2.erode(mask, kernel)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

images = [img, mask, dilation, erosion, opening, closing]
plt.figure(figsize=(15,15))
for i, im in enumerate(images):
    plt.subplot(2, 3, i+1)
    plt.imshow(im, 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(title[i])
plt.show()