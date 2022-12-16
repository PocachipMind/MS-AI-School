import random
import cv2
import albumentations as A
from torch.utils.data import Dataset
import json


# json값 확인
# with open('./instances_default.json', 'r') as f:

#     json_data = json.load(f)

# for i in json_data:
#     print(i)
#     print(json_data[i])


# json xml -> 커스텀 데이터셋 -> 첫번째 실습 코드 결과 제출

# 함수에 쓰일 상수변수
BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # white


# json 받아서 출력 클래스 

class json_custom(Dataset):

    # 주어진 json 파일로부터 데이터를 받음.
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.json_data = json.load(f)
        self.category_id_to_name = dict()
        for i in range(len(self.json_data['categories'])):
            self.category_id_to_name[self.json_data['categories'][i]['id']] = self.json_data['categories'][i]['name']

    # 반환값 : filename, (width, height)
    def __getitem__(self, index):
        # 인덱스값이 csv데이터의 양만큼만 올라가도록함.
        if index >= self.__len__():
            raise IndexError
        
        file_name = self.json_data['images'][index]['file_name']
        w, h = self.json_data['images'][index]['width'], self.json_data['images'][index]['height']

        return file_name, (w,h) # 파일 명이 곧 경로( 같은 폴더에 있으므로 )

    # 길이 반환
    def __len__(self):
        return len(self.json_data['images'])

    # 이 클래스가 가진 모든 이미지 출력
    def all_image_visualize_bbox(self,color=BOX_COLOR, thickness=2):
        for i in range(self.__len__()):
            img = cv2.imread(self.json_data['images'][i]['file_name']).copy()

            for j in range(len(self.json_data['annotations'])):
                bbox = self.json_data['annotations'][j]['bbox']
                category_id = self.json_data['annotations'][j]['category_id']

                class_name = self.category_id_to_name[category_id]

                x_min, y_min, w, h = bbox
                x_min, x_max, y_min, y_max = int(x_min), int(
                    x_min + w), int(y_min), int(y_min + h)

                cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                            color=color, thickness=thickness)
                cv2.putText(img, text=class_name, org=(x_min, y_min-15),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=color,
                            thickness=thickness)
        cv2.imshow("test", img)
        cv2.waitKey(0)

    # 들어온 transform으로 클래스가 가진 이미지를 변경하고 출력
    def all_image_transform(self, transform=None , color=BOX_COLOR, thickness=2):
        for i in range(self.__len__()):
            img = cv2.imread(self.json_data['images'][i]['file_name']).copy()

            bboxes = []
            category_ids = []

            for j in range(len(self.json_data['annotations'])):
                bboxes.append(self.json_data['annotations'][j]['bbox'])
                category_ids.append(self.json_data['annotations'][j]['category_id'])

            transformed = transform(image=img, bboxes=bboxes, category_ids=category_ids)


            for bbox, category_id in zip(transformed['bboxes'], transformed['category_ids']):
                class_name = self.category_id_to_name[category_id]
                x_min, y_min, w, h = bbox
                x_min, x_max, y_min, y_max = int(x_min), int(
                    x_min + w), int(y_min), int(y_min + h)

                cv2.rectangle(transformed['image'], (x_min, y_min), (x_max, y_max),color=color, thickness=thickness)
                cv2.putText(transformed['image'], text=class_name, org=(x_min, y_min-15),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=color,
                            thickness=thickness)

            cv2.imshow("test", transformed['image'])
            cv2.waitKey(0)



######################################################################

# 테스트 코드

# 만든 클래스에 json 파일 입력
test = json_custom('./instances_default.json')


# json속 모든 이미지의 filename , (w,h) 출력
for i in test:
    print(i)

 #csv속 모든 이미지 정보를 통해 Box치고 출력
test.all_image_visualize_bbox()


# 적용할 transform
transfor = A.Compose([
    A.RandomSizedBBoxSafeCrop(width=448, height=336, erosion_rate=0.2),
    A.HorizontalFlip(p=1),
    A.RandomRotate90(p=1),
    A.MultiplicativeNoise(multiplier=0.5, p=1)
    # A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=1)
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

# test클래스의 모든 이미지 넣은 transform으로변경후 출력
test.all_image_transform(transfor)