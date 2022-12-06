'''
이미지 이진화(임계처리)는 어떤 값보다 큰 값을 가진 픽셀을 흰색으로 만들고 작은 값을 가진 픽셀은 검은색으로 만드는 과정이다.
더 고급 기술은 적응적 이진화(Adaptive Thresholding)로, 픽셀의 임곗값이 주변 픽셀의 강도에 의해 결정된다.
이는 이미지 안의 영역마다 빛 조건이 달라질 때 도움이 된다.
'''
# 이진화를 할때 그레이로 해야함. 컬러로 하면 적용이 안됨

import cv2
from utils import image_show

# 이미지 경로
image_path = "./cat.png"

# 이미지 읽기
image = cv2.imread(image_path)

# 이미지 이진화
image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
max_output_value = 255 # 출력 픽셀 강도의 최대값
neighborhood_size = 99
subtract_from_mean = 10 # 평균값 ( 높으면 하얀색이 비중 높아짐 )

image_binary = cv2.adaptiveThreshold(image_gray,
                                        max_output_value,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, neighborhood_size,
                                        subtract_from_mean) # cv2.THRESH_BINARY_INV 넣으면 반전됨

image_show(image_binary,"Image_binary")

'''
adaptiveThreshold() 함수에는 네 개의 중요한 매개변수가 있다.
• max_output_value : 출력 픽셀 강도의 최댓값 저장
• cv2.ADAPTIVE_THRESH_GAUSSIAN_C : 픽셀의 임곗값을 주변 픽셀 강도의 가중치 합으로 설정. 가중치
는 가우시안 윈도우에 의해 결정
• cv2.ADAPTIVE_THRESH_MEAN_C : 주변 픽셀의 평균을 임곗값으로 설정
'''