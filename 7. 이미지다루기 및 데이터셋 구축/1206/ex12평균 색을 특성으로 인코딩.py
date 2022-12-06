'''
이미지의 각 픽셀은 여러 컬러 채널(빨간, 초록, 파랑)의 조합으로 표현되며, 
채널의 평균값을 계산하여 이미지의평균 컬러를 나타내는 세 개의 컬럼 특성을 만듭니다.
'''
# 평균색 특성 인코딩
import cv2
import numpy as np

image = cv2.imread('./cat.png')

channels = cv2.mean(image) # 각 채널의 평균 계산
print("channels :",channels)
# channels : (205.8482456140351, 205.4207560568087, 207.63455722639935, 0.0)


# 파랑과 빨강을 바꿉니다 ( BGR -> RGB )
observation = np.array([(channels[2], channels[1], channels[0])])
print("observation : ",observation)
# observation :  [[207.63455723 205.42075606 205.84824561]]