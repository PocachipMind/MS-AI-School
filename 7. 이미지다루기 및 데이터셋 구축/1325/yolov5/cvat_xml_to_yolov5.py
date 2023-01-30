# cvat xml to yolo

import os
import glob
import cv2
from xml.etree.ElementTree import parse

# xml 1 ~ 5
def find_xml_file(xml_folder_path) :
    all_root = []
    for (path, dir, files) in os.walk(xml_folder_path) :
        for filename in files :
            ext = os.path.splitext(filename)[-1]
            # ext -> .xml
            if ext == ".xml" :
                root = os.path.join(path, filename)
                # ./xml_data/test.xml
                all_root.append(root)
            else :
                print("no xml files.....")
                break

    return all_root

xml_folder_dir = "./xml_data"
xml_paths = find_xml_file(xml_folder_dir)
# test = glob.glob(os.path.join(xml_folder_path, "*.xml"))

# 기존의 0 : "big bus" 형식의 dict에가 label_dict라면,
# label_dict = {v:k for k,v in label_dict.items()} 로도 변환 가능함.
label_dict = {"big bus" : 0 , "big truck" : 1 , "bus-l-":2, "bus-s-":3, "car":4, "mid truck":5,
              "small bus":6, "small truck":7, "truck-l-":8, "truck-m-":9, "truck-s-":10, "truck-xl-":11}

for xml_path in xml_paths :
    tree = parse(xml_path)
    root = tree.getroot()
    img_metas = root.findall("image")
    for img_meta in img_metas :
        # xml image name
        image_name = img_meta.attrib["name"]
        # print(image_name)
        # image_name >> aditganteng_mp4-165_jpg.rf.976ae8b8ed6d79aab3f9566dba1f4645.jpg

        # Box META
        box_metas = img_meta.findall("box")
        img_width = int(img_meta.attrib["width"])
        img_height = int(img_meta.attrib["height"])
        for box_meta in box_metas :
            box_label = box_meta.attrib["label"]
            box = [int(float((box_meta.attrib["xtl"]))), int(float((box_meta.attrib["ytl"]))),
                   int(float((box_meta.attrib["xbr"]))), int(float((box_meta.attrib["ybr"])))]

            # 정규화 하기
            yolo_x = round(((box[0] + box[2])/2)/img_width, 6)
            yolo_y = round(((box[1] + box[3])/2)/img_height, 6)
            yolo_w = round((box[2] - box[0])/img_width, 6)
            yolo_h = round((box[3] - box[1])/img_height, 6)

            print("yolo xywh" , yolo_x, yolo_y, yolo_w, yolo_h)
            image_name_temp = image_name.replace(".jpg", ".txt")

            # txt file save folder
            os.makedirs("./cvat_xml_to_yolo_txt", exist_ok=True)

            # label
            label = label_dict[box_label]

            # txt save
            with open(f"./cvat_xml_to_yolo_txt/{image_name_temp}", 'a') as f :
                f.write(f"{label} {yolo_x} {yolo_y} {yolo_w} {yolo_h} \n")


