# rotate(이미지, 각도) : rotate 함수의 첫번째 매개변수에는 회전시킬 이미지를, 두번째 매개변수에는 회전각도를 입력해주면 됩니다

import cv2

# 이미지 경로
image_path = "./cat.png"

# 이미지 읽기
image = cv2.imread(image_path)

img90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) # 시계방향으로 90도 회전
img180 = cv2.rotate(image, cv2.ROTATE_180) # 180도 회전
img270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) # 반시계방향으로 90도 ( 시계방향으로 270도 )

# 회전은 크기를 변화시키지는 않습니다.
print(image.shape)
print(img90.shape)

cv2.imshow("original image", image)
cv2.imshow("rotate_90", img90)
cv2.imshow("rotate_180", img180)
cv2.imshow("rotate_270", img270)

cv2.waitKey(0) 

# 이미지 좌우 및 상하 반전
# Flip(이미지, key) : key 1은 좌우 0은 상하

dst_temp1 = cv2.flip(image,1)
dst_temp2 = cv2.flip(image,0)

cv2.imshow("reverse left and right image", dst_temp1) # 좌우반전 이미지
cv2.imshow("upside down image", dst_temp2) # 상하반전 이미지
cv2.waitKey(0) 