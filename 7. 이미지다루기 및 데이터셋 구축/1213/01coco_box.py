import os
import json
import cv2

# json path
json_path = './annotations/instances_default_box.json'

# json 파일 읽기

with open(json_path,'r') as f:
    coco_info = json.load(f)

# print(coco_info)

assert len(coco_info) > 0, "파일 읽기 실패"

# 카테고리 정보 수집
categories = dict()

for category in coco_info['categories']:
    print(category) # {'id': 1, 'name': 'kiwi', 'supercategory': ''}
    categories[category['id']] = category['name']

print("categories info >> ", categories) # categories info >>  {1: 'kiwi'}

# annotation 정보 수집
ann_info = dict()

for annotation in coco_info['annotations']:
    # print(annotation)
    image_id = annotation['image_id']
    bbox = annotation['bbox']
    category_id = annotation['category_id']
    # print(f'image_id : {image_id}, category_id : {category_id}, bbox : {bbox}')

    if image_id not in ann_info:
        ann_info[image_id] = {
            'boxes' : [bbox], 'categories' : [category_id]
        }
    else:
        ann_info[image_id]['boxes'].append(bbox)
        ann_info[image_id]['categories'].append(categories[category_id])

# print(ann_info)

os.makedirs('./coco_box', exist_ok=True)
for image_info in coco_info['images']:
    # print(image_info)
    filename = image_info['file_name']
    width = image_info['width']
    height = image_info['height']
    img_id = image_info['id']
    # print(filename, width, height, img_id)
    file_path = os.path.join('./images', filename)
    
    # 이미지 가져오기 위한 처리
    print('file path =',file_path)
    img = cv2.imread(file_path)
    try:
        annotation = ann_info[img_id]
    except KeyError:
        continue

    ## box category
    for bbox, category in zip(annotation['boxes'], annotation['categories']):
        # print(bbox, category)
        x1, y1, w, h = bbox
        print(x1, y1, w, h)

        rec_img = cv2.rectangle(img, (int(x1), int(y1)), (int(x1+w), int(y1+h)), (225, 0, 255), 2 )

    cv2.imwrite(f"./coco_box/{filename}", rec_img)
    # cv2.imshow('test', rec_img)
    # cv2.waitKey(0)