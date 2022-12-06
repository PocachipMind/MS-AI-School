# 이미지의 각 픽셀은 여러 컬러 채널(빨간, 초록, 파랑)의 조합으로 표현되며, 채널의 평균값을 계산하여
# 이미지의 평균 컬러를 나타내는 세 개의 컬럼 특성을 만듭니다.

# 컬러 히스토그램 특성 인코딩
import cv2
import matplotlib.pyplot as plt

image_path = './cat.png'

image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
features = [] # 특성 값을 담을 리스트
colors = ('r', 'g', 'b') # 각 컬러 채널에 대해 히스토그램을 계산

# 각 채널을 반복하면서 히스토그램을 계산하고 리스트에 추가
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image_rgb], # 이미지
                             [i], # 채널 인덱스
                             None, # 마스크. ( 없으므로 None )
                             [256], # 히스토그램 크기
                             [0,256] ) # 범위
    plt.plot(histogram, color=channel)
    plt.xlim([0, 256])

plt.show()
