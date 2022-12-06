import cv2

def image_show(image, ti="show"):
    cv2.imshow(ti,image)
    cv2.waitKey(0)