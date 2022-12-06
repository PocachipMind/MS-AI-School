'''
이미지 데이터는 본래 개별 원소로 이루어진 행렬의 집합이다. 
여기서 개별 원소는 픽셀(pixel)이라고 할 수 있으며 개별 원소의 값은 픽셀의 강도라고 할 수 있다. 
그리고 픽셀의 강도는 0(검정)부터 255(흰색) 사이의 범위를 가지고 있다
'''

import cv2

img_path = './cat.png'
img = cv2.imread(img_path)

print("이미지 타입 :", type(img))
print("이미지 크기 :", img.shape)
'''
이미지 타입: <class 'numpy.ndarray'>
이미지 크기 : (399, 600, 3)
'''

# 높이, 넓이, 채널
h , w, _ = img.shape # 이렇게 자주씁니다.
print(f"이미지 높이 {h}, 이미지 넓이 {w}") # 이미지 높이 399, 이미지 넓이 600

# 이미지 읽기
cv2.imshow("image show", img)
cv2.waitKey(0)