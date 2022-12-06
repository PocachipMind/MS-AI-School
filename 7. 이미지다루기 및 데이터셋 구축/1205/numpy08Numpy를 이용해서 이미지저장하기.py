import numpy as np
import cv2
# pip install opencv-python==4.5.5.62
# 4.5.5.62 버전을 쓰는걸 추천. 오히려 버그가 더 적음!

x = cv2.imread("./cat.png",0) # 이미지를 흑백으로 불러옵니다
y = cv2.imread("./cat.png",1) # 이미지를 컬러으로 불러옵니다
#배열 형태로 들어갑니다.

cv2.imshow("cat image show gray", x)
cv2.imshow("cat image show", y)
cv2.waitKey(0) # 창이 떴다가 꺼지지 않게 하는 코드

# cv2.resize를 통해 창 크기 변경
img = cv2.resize(x, (200, 200))
cv2.imshow("cat image show resize", img)
cv2.waitKey(0) # 창이 떴다가 꺼지지 않게 하는 코드

# 하나의 파일 save는 np.save('~~',x)


# 여러개 파일 save 하는 방법
np.savez("./image.npz", array1=x, array2=y)

# 압축 방법
np.savez_compressed("./image_compressed.npz", array1=x, array2=y)

# npz 데이터 로드
data = np.load("./image_compressed.npz")

result1 = data['array1']
result2 = data['array2']

#cv2 통해서 데이터 읽기
cv2.imshow("result01", result1)
cv2.waitKey(0) # 비디오일 경우 1이 맞습니다 0이면 멈춰요