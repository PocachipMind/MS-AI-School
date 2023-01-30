# 오토 라벨링 코드

import torch
import os
import glob
import cv2
import xml.etree.ElementTree as ET
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model call
model = torch.hub.load('ultralytics/yolov5', 'custom', path="./runs/train/exp_0125/weights/best.pt")

# inference Settings
model.conf = 0.5 # NMS confidence threshold ( 얼마 이상 점수 표시할지)
model.iou = 0.45 # NMS IoU threshold ( 곂치는거 )
# model.cuda() # GPU
model.to(device)

# image loader
image_dir = "./dataset/test/images/"
image_path = glob.glob(os.path.join(image_dir, "*.jpg"))
label_dict = {0:"big bus", 1:"big truck", 2:"bus-l-", 3:"bus-s-",4:"car",
              5:"mid truck", 6:"small bus", 7:"small truck", 8:"truck-l-", 9:"truck-m-",
              10:"truck-s-",11:"truck-xl-"}

# 맨위 annotation 만들어 준겁니다.
tree = ET.ElementTree()
root = ET.Element("annotations")
"""
<annotations>
</annotations>
"""
seen_count = 0

for img_path in image_path :
    # Image
    img = cv2.imread(img_path)

    # Inference
    results = model(img, size=640)

    # Results
    bbox = results.xyxy[0]

    # image name
    image_name = os.path.basename(img_path)

    # image w h
    h, w, c = img.shape

    # xml fix code
    # <image id="0" name="adit_mp4-1002_jpg.rf.5e4018e963af1251b3f7e6fd487c479e.jpg" width="640" height="480">
    xml_frame = ET.SubElement(root, "image" , id="%d"%seen_count, name=image_name,
                              width="%d"%w, height="%d"%h)
    """
    <annotations>
        <image id="0" name="adit_mp4-1002_jpg.rf.5e4018e963af1251b3f7e6fd487c479e.jpg" width="640" height="480">
        </image>
    </annotations>
    """

    for box in bbox :
        # <box label="car" occluded="0" source="manual" xtl="346.68"
        # ytl="325.63" xbr="427.46" ybr="404.08" z_order="0"> </box>
        # box
        """
        <annotations>
            <image id="0" name="adit_mp4-1002_jpg.rf.5e4018e963af1251b3f7e6fd487c479e.jpg" width="640" height="480">
                <box></box>
                <box></box>
                <box></box>
                <box></box>
            </image>
            <image id="0" name="adit_mp4-1002_jpg.rf.5e4018e963af1251b3f7e6fd487c43249e.jpg" width="640" height="480">
                <box></box>
                <box></box>
                <box></box>
                <box></box>
            </image>
        </annotations>
        """
        # 박스 요소 꺼내기
        x1 = box[0].item()
        y1 = box[1].item()
        x2 = box[2].item()
        y2 = box[3].item()
        xtl = str(round(x1,3)) # x top left
        ytl = str(round(y1,3))
        xbr = str(round(x2,3)) # x bottom right
        ybr = str(round(y2,3))

        # clss
        clss_number = box[5].item()
        clss_number_int = int(clss_number)
        labels = label_dict[clss_number_int]

        # sc number
        sc = box[4].item()

        # bbox xml
        #  xtl="346.68" ytl="325.63" xbr="427.46" ybr="404.08" z_order="0"
        ET.SubElement(xml_frame, "box", label=labels, occluded="0", source="manual",
                      xtl=xtl, ytl=ytl, xbr=xbr, ybr=ybr, z_order="0")

    seen_count +=1

tree._setroot(root)
tree.write("test.xml", encoding="utf-8")

