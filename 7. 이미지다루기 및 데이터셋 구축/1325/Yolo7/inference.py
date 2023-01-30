import torch
import os
import glob
import cv2
from hubconf import custom

# device setting
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model call
model = custom(path_or_model='./runs/train/exp2/weights/best.pt')
model.conf = 0.6  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.to(DEVICE)

# image path list
image_path_list = glob.glob(os.path.join("./dataset/test/images", "*.jpg"))

for i in image_path_list :
    image_path = i

    # cv2 image read
    image = cv2.imread(image_path)

    # model input
    output = model(image, size=640)
    bbox_info = output.xyxy[0]
    for bbox in bbox_info :
        x1 = int(bbox[0].item())
        y1 = int(bbox[1].item())
        x2 = int(bbox[2].item())
        y2 = int(bbox[3].item())

        score = round(bbox[4].item(),4)
        label_number = int(bbox[5].item())
        print(x1, y1, x2, y2, score, label_number)
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,255),2)

    cv2.imshow("test",image)
    cv2.waitKey(1)
