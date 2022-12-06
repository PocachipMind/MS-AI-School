'''
배경을 제거하고자 하는 전경 주위에 사각형 박스를 그리고 그랩컷(grabCut) 알고리즘을 적용하여 배경을제거한다.
grabCut의 경우 잘 작동하더라도 여전히 이미지에 제거하지 못한 배경이 발생할 수 있다. 
이렇게 제거 되지 못한 부분은 다시 적용하여 제거할 수 있지만 실전에서 수 천장의 이미지를 수동으로 고치는 것은 불가능한 일이므로 
머신러닝을 적용한다거나 할 때도 일부러 noise를 적용하는 것처럼 일부 배경이 남아있는 것을 수용하는 것이 좋다.
'''

# 배경제거
import cv2
import numpy as np
from utils import image_show

# 이미지 경로
image_path ='./pizza.png'
# 이미지 읽기
image = cv2.imread(image_path)

# 사각형 좌표 : 시작점의 x, y , 넓이, 높이
rectangle = (0, 0, 400, 400)

# 초기 마스크 생성
mask = np.zeros(image.shape[:2], np.uint8)

# grabCut에 사용할 임시 배열 생성
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# grabCut 실행
cv2.grabCut(image,    # 원본 이미지
            mask,     # 마스크
            rectangle,
            bgdModel, # 사각형
            fgdModel, # 배경을 위한 임시 배열
            5,        # 반복 횟수
            cv2.GC_INIT_WITH_RECT) # 사각형을 위한 초기화

# 배경인 곳은 0, 그 외에는 1로 설정한 마스크 생성
mask_2 = np.where((mask == 2) | ( mask == 0), 0, 1).astype('uint8')

# 이미지에 새로운 마스크를 곱해서 -> 배경을 제거
image_nobg = image * mask_2[:,:,np.newaxis]
image_show(image_nobg)

# 사각형 좌표 : 시작점의 x, y , 넓이, 높이
# image = cv2.rectangle(image, (90, 30),(150,150),(255, 0, 255, 2)) # 이미지에 사각형 그리기
# image_show(image)