'''
dataset
    - train
        - cat
            - cat1.jpg ...
        - dog
            - dog1.jpg ...
    - val
        - cat
        - dog
    - test
        - cat
        - dog
'''

from torch.utils.data.dataset import Dataset
import os
import glob
from PIL import Image

label_dit = { 'cat' : 0 , 'dog' : 1}

class cat_dog_mycustomdataset(Dataset):
    def __init__(self, data_path):
        # csv folder 읽기, 변환 할당, 데이터 필터링 등 과 같은 초기 논리가 발생
        # data_path -> ./dataset/train
        # train -> ./dataset/train
        # val -> ./dataset/val
        # test -> ./dataset/test
        self.all_data_path = glob.glob(os.path.join(data_path,'*',"*.jpg"))
        
    
    def __getitem__(self, index):
        # 데이터 레이블 반환 image, label
        image_path = self.all_data_path[index]
        img = Image.open(image_path).convert('RGB')
        label_temp = image_path.split('/')
        # label_temp 출력값 >> ['.', 'dataset', 'train\\cat\\cat.1.jpg']
        label = label_dit[label_temp[2].split('\\')[1]]

        return img, label
    
    def __len__(self):
        # 전체 데이터의 길이 반환
        return 

test = cat_dog_mycustomdataset("./dataset/train/")

for i in test:
    pass