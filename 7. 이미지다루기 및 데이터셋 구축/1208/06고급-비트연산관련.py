import cv2
import numpy as np

# ex-01
img_rectangle = np.ones((400,400), dtype='uint8')
cv2.rectangle(img_rectangle, (50,50),(300,300),(255,255,255),-1)
cv2.imshow("image show",img_rectangle)
cv2.waitKey(0)

# ex-02
img_circle = np.ones((400,400), dtype='uint8')
cv2.circle(img_circle, (300,300),70,(255,255,255),-1)
cv2.imshow("image show",img_circle)
cv2.waitKey(0)

# ex -03
bitwiseAnd = cv2.bitwise_and(img_rectangle, img_circle)
cv2.imshow("image bitwiseAnd", bitwiseAnd)
cv2.waitKey(0)

bitwiseOr = cv2.bitwise_or(img_rectangle, img_circle)
cv2.imshow("image bitwiseOr", bitwiseOr)
cv2.waitKey(0)

bitwiseXor = cv2.bitwise_xor(img_rectangle, img_circle)
cv2.imshow("image bitwiseXor", bitwiseXor)
cv2.waitKey(0)

rec_not = cv2.bitwise_not(img_rectangle)
cv2.imshow('rectangle not ', rec_not)
cv2.waitKey(0)

cir_not = cv2.bitwise_not(img_circle)
cv2.imshow('circle not ', cir_not)
cv2.waitKey(0)