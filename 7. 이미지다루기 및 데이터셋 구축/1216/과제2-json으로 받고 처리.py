# 이미지 사이즈 -> 바운딩 위치
import cv2
import numpy as np
from torch.utils.data import Dataset
import json

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

    def cvTest(self):
        for i in range(self.__len__()):
            image = cv2.imread(self.json_data['images'][i]['file_name']).copy()
            y_ = image.shape[0]
            x_ = image.shape[1]

            target_size = 400
            x_scale = target_size / x_
            y_scale = target_size / y_
            print("x_scle >> ", x_scale, "y_scle >> ", y_scale)

            img = cv2.resize(image, (target_size, target_size))
            bboxes = []

            for j in range(len(self.json_data['annotations'])):
                bboxes.append(self.json_data['annotations'][j]['bbox'])

            for boxs in bboxes:
                x_min, y_min, w, h = boxs

                # xywh to xyxy
                x_min, x_max, y_min, y_max = int(x_min), int(
                    x_min + w), int(y_min), int(y_min + h)

                x1 = int(np.round(x_min * x_scale))
                y1 = int(np.round(y_min * y_scale))
                x2 = int(np.round(x_max * x_scale))
                y2 = int(np.round(y_max * y_scale))

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

            cv2.imshow("test", img)
            cv2.waitKey(0)


if __name__ == '__main__':
    test = json_custom('./instances_default.json')
    test.cvTest()
