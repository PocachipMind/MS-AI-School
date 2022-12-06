# 형태학적 변환 팽창 과 침식
# python에서 확장, 침식 실험 예시

import cv2
import matplotlib.pyplot as plt

# image loading and input image -> gray
img = cv2.imread('./Billiards.png', cv2.IMREAD_GRAYSCALE)

# 임계값 연산자의 출력을 마스크라는 변수에 저장
# 230보다 작으면 모든 값을 흰색 처리/ 230 보다 큰 모든 값은 검은색 처리
_, mask = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)

'''
우리가 사용하는 추가 매개변수는 임계 강도 픽셀 값이며, 이 경우에는 230과 255 값으로 설정됩니다.
이는 230보다 큰 모든 값이 255 값으로 설정됨을 의미합니다. 우리가 사용하는 마지막 매개변수는 유형입니다.
단순히 값을 반전시키는 임계값 알고리즘 THRESH_BINARY_INV입니다
(230보다 작은 모든 값은 흰색이 되고 230보다 큰 모든 값은 검은 색이 됩니다.)
'''

titles =['image', 'mask']
images = [img, mask]

for i in range(2) :
    plt.subplot(1, 2, i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()