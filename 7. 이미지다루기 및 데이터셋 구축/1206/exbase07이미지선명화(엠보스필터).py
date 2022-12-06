'''
# 엠보스 효과
필터의 방향에 따라 사용할 수 있는 여러 필터가 있습니다. 
이 필터의 수직, 수평 또는 대각선 버전을 가질 수 있습니다. 
우리의 경우 수직 이미지를 사용하고 일단 이미지를 필터링하면 매우 낮은 차이를 얻게 됩니다. 
즉, 출력 이미지가 다소 검은색이 됩니다. 
따라서 각 픽셀에 상수 128을 추가하고 결과 이미지를 회색으로 얻습니다.
'''
import cv2
import numpy as np
from utils import image_show

image = cv2.imread('./car.png' )

# 엠보싱 효과
filter1 = np.array([[0, 1, 0], 
                    [0, 0, 0], 
                    [0,-1, 0]])

filter2 = np.array([[-1,-1, 0], 
                    [-1, 0, 1], 
                    [ 0, 1, 1]])


emboss_img1 = cv2.filter2D(image,-1, filter1)
emboss_img2 = cv2.filter2D(image,-1, filter2)
# 어두우므로 회색추가
emboss_img1 = emboss_img1 + 128 # 128이 회색
emboss_img2 = emboss_img2 + 128

# 비교해보기
cv2.imshow("emboss_img1", emboss_img1)
image_show(emboss_img2, "emboss_img2")