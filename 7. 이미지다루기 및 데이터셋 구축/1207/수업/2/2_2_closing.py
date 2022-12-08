'''
closing : dilation -> erosion (to detect object edge line)
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../Billiards.png', cv2.IMREAD_GRAYSCALE)

_, mask = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)

# datatype : int, float
kernel = np.ones((3, 3), np.uint8)

N = 1
idx = 1
plt.figure(figsize=(15, 15))
for i in range(1, N + 1):
    erosion = cv2.dilate(mask, kernel, iterations=i)
    closing = cv2.erode(erosion, kernel, iterations=i)
    f_closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=i)

    plt.subplot(N, 2, idx)
    idx += 1
    plt.imshow(closing, 'gray')
    plt.title(f'{i} manual opening')

    plt.subplot(N, 2, idx)
    plt.imshow(f_closing, 'gray')
    plt.title(f'{i} function opening')
    idx += 1
plt.tight_layout()
plt.show()
