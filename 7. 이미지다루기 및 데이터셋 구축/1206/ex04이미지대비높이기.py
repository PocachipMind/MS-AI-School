# 히스토그램 평활화(Histogram Equalization)은 객체의 형태가 두드러지도록 만들어주는 이미지 처리 도구이며, OpenCV에서는 equalizeHist() 메소드를 통해 적용할 수 있다.
#컬러 이미지의 경우 먼저 YUV 컬러 포맷으로 변환해야 한다. Y는 루마 또는 밝기이고 U와 V는 컬러를 나타낸다. 
#변환한 뒤에 위와 동일하게 equlizeHist() 메소드를 적용하고 다시 RGB 포맷으로 변환 후 출력한다.

import cv2
import matplotlib.pyplot as plt

image_path = "./cat.png"

# 흑백 이미지 대비 높이기
image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image_enhanced = cv2.equalizeHist(image_gray)

# plot
fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].imshow(image_gray, cmap='gray')
ax[0].set_title("Original Image")
ax[1].imshow(image_enhanced, cmap='gray')
ax[1].set_title("Enhanced Image")

plt.show()

##########################################################

# 컬러 이미지 대비 높이기
# 방법 : RGB -> YUV 컬러 포맷으로 변환 -> equlizeHist() -> RGB
# BGR
image_bgr = cv2.imread(image_path) # cv2.IMREAD_COLOR 넣어도댐

# RGB 타입으로 변환
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # matplot에서는 BGR로 받아들이므로 RGB로 변환하는 과정임.

# YUV 컬러 포멧으로 변환
image_yuv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)

# 히스토그램 평활화 적용
image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])

# RGB로 변경
image_rgb_temp = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

# plot
fig, ax = plt.subplots(1, 2, figsize=(12,8))
ax[0].imshow(image_rgb)
ax[0].set_title("Original Image")
ax[1].imshow(image_rgb_temp)
ax[1].set_title("Enhanced Color Image")

plt.show()