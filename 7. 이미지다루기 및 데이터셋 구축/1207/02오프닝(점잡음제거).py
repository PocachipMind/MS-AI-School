'''
opening : erosion -> dilation (to delete dot noise)
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

img =cv2.imread("./Billiards.png", cv2.IMREAD_GRAYSCALE)

_, mask = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)

# datatype : int, float
kernel = np.ones((3,3), np.uint8)

erosion = cv2.erode(mask, kernel, iterations=1)
opening = cv2.dilate(erosion, kernel, iterations=1)

plt.subplot(1,2,1)
plt.imshow(opening, 'gray')
plt.title('manual opening')

f_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
plt.subplot(1,2,2)
plt.imshow(f_opening, 'gray')
plt.title('function opening')
plt.show()

'''
오프닝과 이로션 하고 딜레이션한것이구
클로징은 딜레이션하고 이로션 하는것 - 윤곽을 잘 잡기위해 쓰여짐
'''

# 참고 : https://webnautes.tistory.com/1257
