import cv2

large_img = cv2.imread('./ex_image.png')
watermark = cv2.imread('./ex_image_logo.png')

small_img = cv2.resize(watermark, (300,300))
# cv2.imshow('test', small_img)
# cv2.waitKey(0)

x_offset = 30
y_offset = 170

x_end = x_offset + small_img.shape[0]
y_end = y_offset + small_img.shape[1]

crop = large_img[y_offset:y_end, x_offset:x_end]
large_img[y_offset:y_end, x_offset:x_end] = small_img
cv2.imshow('test',large_img)
cv2.waitKey(0)