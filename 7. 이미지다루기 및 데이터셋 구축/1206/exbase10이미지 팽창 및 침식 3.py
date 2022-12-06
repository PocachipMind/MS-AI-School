import cv2
import matplotlib.pyplot as plt
import numpy as np

img_gray = cv2.imread('./Billiards.png', cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY_INV)

# 3x3 커널 생성
kernel = np.ones((3,3), np.uint8) 

dilation = cv2.dilate(mask, kernel, iterations=10)

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
팽창을 두 번 이상 적용할 수 있는지 물어보셨을 것입니다. 예, 가능합니다. 이를 위해 반복 횟수라는 매개 변수를 사용할 수 있습니다. 
예를 들어, 이 매개변수를 10으로 설정할 수 있습니다. 이는 확장 프로세스가 연속적으로 10번 반복됨을 의미합니다. 
결과 이미지에서 훨씬 더 많은 검은색 영역이 이미지에서 사라진 반면 흰색 영역은 확장된 것을 볼 수 있습니다.
'''