import numpy as np
import random
import cv2
import albumentations as A

keypoints = [
    (100, 100, 50, np.pi/4.0),
    (720, 410, 50, np.pi/4.0),
    (1100, 400, 50, np.pi/4.0),
    (1700, 30, 50, np.pi/4.0),
    (300, 650, 50, np.pi/4.0),
    (1570, 590, 50, np.pi/4.0),
    (560, 800, 50, np.pi/4.0),
    (1300, 750, 50, np.pi/4.0),
    (900, 1000, 50, np.pi/4.0),
    (910, 780, 50, np.pi/4.0),
    (670, 670, 50, np.pi/4.0),
    (830, 670, 50, np.pi/4.0),
    (1000, 670, 50, np.pi/4.0),
    (1150, 670, 50, np.pi/4.0),
    (820, 900, 50, np.pi/4.0),
    (1000, 900, 50, np.pi/4.0),
]

KEYPOINT_COLOR = (0, 255, 0)  # green


def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=2):
    image = image.copy()
    for (x, y, s, a) in keypoints:
        print(x, y, s, a)
        cv2.circle(image, (int(x), int(y)), diameter, color, -1)

        x0 = int(x) + s * np.cos(a)
        y0 = int(y) - s * np.sin(a)
        cv2.arrowedLine(image, (int(x), int(y)), (int(x0), int(y0)), color, 2)

    cv2.imshow("test", image)
    cv2.waitKey(0)


image = cv2.imread("./keypoints_image.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


transform = A.Compose([
    A.ShiftScaleRotate(p=1),
], keypoint_params=A.KeypointParams(format='xysa', angle_in_degrees=False))

transformed = transform(image=image, keypoints=keypoints)
vis_keypoints(transformed['image'], transformed['keypoints'])
# vis_keypoints(image, keypoints)
