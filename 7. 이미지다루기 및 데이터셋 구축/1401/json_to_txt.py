import os
import glob
import cv2
import json

def get_json_paths(root_path):
    # root_path 경로 하위에 있는 모든 파일 경로 탐색
    paths = []
    for (path, dir, files) in os.walk(root_path):
        for file in files:
            if file.split('.')[-1] != 'json':
                continue

            image_path = os.path.join(path, file)
            paths.append(image_path)
    return paths

json_folder_dir = "./xml_data"
json_paths = get_json_paths(json_folder_dir)


label_dict = {"garbage_bag" : 0 , "sit_board" : 1 , "street_vendor":2, "food_truck":3, "banner":4, "tent" :5,
              "smoke":6, "flame":7, "pet":8, "bench":9, "park_pot":10, "trash_can" : 11 , "rest_area" : 12, "toilet" : 13 ,
              "street_lamp" : 14, "park_info" :15 }

for json_path in json_paths:
    try:
        # json 읽기
        with open(json_path, 'r', encoding='utf8') as f:
            # json load
            json_file = json.load(f)

        annotations = json_file['annotations']
        image_info = json_file['images']

        # yolo width heigth image size getting
        img_width = image_info['width']
        img_height = image_info['height']

        # text file name
        img_name = image_info['ori_file_name']
        img_name = img_name.replace('.jpg', ".txt")

        for annotation in annotations:
            object_class = annotation['object_class']
            if object_class not in label_dict:
                continue  # 비석 울타리 제외
            bbox = annotation['bbox']
            x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]

            # voc -> yolo 좌표 얻기
            yolo_x = (x1 + x2) / 2 / img_width
            yolo_y = (y1 + y2) / 2 / img_height
            yolo_w = (x2 - x1) / img_width
            yolo_h = (y2 - y1) / img_height

            # 라벨 얻기
            label_number = label_dict[object_class]

            print(yolo_x, yolo_y, yolo_w, yolo_h, label_number)

            # yolo 좌표와 라벨을 텍스트 파일로 쓰기
            os.makedirs("./yolo_labels", exist_ok=True)
            with open(f"./yolo_labels/{img_name}", 'a') as f:
                f.write(f"{label_number} {yolo_x} {yolo_y} {yolo_w} {yolo_h} \n")

    except Exception as e:
        print(e)
