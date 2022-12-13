import json
import pandas as pd

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

columns = ['file_name', 'x1', 'y1','w','h']
df = pd.DataFrame(columns=columns)
data = dict()
for i in columns:
    data[i] = []

for image_info in coco_info['images']:
    filename = image_info['file_name']
    width = image_info['width']
    height = image_info['height']
    img_id = image_info['id']

    try:
        annotation = ann_info[img_id]
    except KeyError:
        continue

    ## box category
    for bbox, category in zip(annotation['boxes'], annotation['categories']):

        x1, y1, w, h = bbox
        # print(x1, y1, w, h)

        data['file_name'].append(filename)
        data['x1'].append(x1)
        data['y1'].append(y1)
        data['w'].append(w)
        data['h'].append(h)

for i in columns:
    df[i] = data[i]

print(df)
df.to_csv('./file_and_box_point.csv')