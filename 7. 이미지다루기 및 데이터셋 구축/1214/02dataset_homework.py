from torch.utils.data.dataset import Dataset
import pandas as pd
import os
import glob

class MyCustomDataset(Dataset): # Dataset 쓸 때 상속을 꼭 받아야 함.
    def __init__(self, path):
        self.data_path = path
        self.data2csv = pd.read_csv(path)
        self.data_len = len(self.data2csv)

        # print(self.data2csv)
        # print(len(self.data2csv))

        self.columns_of_data = self.data2csv.columns
        # Index(['Unnamed: 0', 'file_name', 'x1', 'y1', 'w', 'h'], dtype='object')


    def __getitem__(self, index):
        # 인덱스값이 csv데이터의 양만큼만 올라가도록함.
        if index >= self.data_len:
            raise IndexError

        file_name = self.data2csv[self.columns_of_data[1]][index]

        bbox = []
        for i in range(4):
            bbox.append(self.data2csv[self.columns_of_data[i+2]][index])
        
        # 튜플로 반환하려고 한다면
        # bbox = tuple(bbox)

        return file_name, bbox

    def __len__(self):
        return self.data_len

temp = MyCustomDataset('./file_and_box_point.csv') # 파일 경로 입력

for i in temp:
    print(i)
