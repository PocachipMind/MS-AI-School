'''
이미지를 자르고 싶을 경우 배열 슬라이싱을 이용하여 원하는 부분만 crop할 수 있다.
'''
import cv2

def image_show(image, ti="show"):
    cv2.imshow(ti,image)
    cv2.waitKey(0)

image_path = './cat.png'

# 이미지 읽기
image = cv2.imread(image_path)

# 이미지 크롭 [시작 : 끝 : 단계]
#image_crop = image[10:, :300]
#image_show(image_crop)

# 고양이얼굴 맞춰서 해보기!
image_crop = image[50:350, 100:430]
image_show(image_crop)


# 이미지 저장 코드도 작성
cv2.imwrite("./cat_face.png", image_crop)