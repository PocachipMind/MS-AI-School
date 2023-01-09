import os
import glob
from sklearn.model_selection import train_test_split
import cv2


path = "./data"

all_labels = os.listdir(path)


# 1. 이미지 폴더 가져오기

for i in all_labels:
    globals()["img_"+i+"_path"] = f"./data/{i}"
    globals()["img_"+i] = glob.glob(os.path.join(globals()["img_"+i+"_path"], "*.png"))

'''
각 라벨별로 생성됨 >
img_cloudy_path = "./data/cloudy"
img_cloudy = glob.glob(os.path.join(img_cloudy_path, "*.jpg"))
'''


# 2. train, val, test 나누기
''' 이런 방법도 있습니다.
import splitfolders
splitfolders.ratio("path","저장경로",ratio=(.8,.1,.1),seed=7777)
'''

for i in all_labels:
    globals()[i+"_train_list"], globals()[i+"_val_list"] = train_test_split(globals()["img_"+i], test_size=0.2, random_state=7)
    globals()[i+"_val_data"], globals()[i+"_test_data"] = train_test_split(globals()[i+"_val_list"], test_size=0.5, random_state=7)

'''
각 라벨별로 생성됨 >
cloudy_train_list, cloudy_val_list = train_test_split(img_cloudy, test_size=0.2, random_state=7)
cloudy_val_data, cloudy_test_data = train_test_split(cloudy_val_list, test_size=0.5, random_state=7)
'''


# 3. 각 폴더로 이미지 옮기기
def data_save(data, mode) :
    for path in data :
        # image name
        image_name = os.path.basename(path)
        image_name = image_name.replace(".jpg","")

        # 0. 폴더명 구하기
        folder_name = path.split("\\")
        # print(folder_name)
        # ['./data/cloudy', 'train_13464.jpg']
        folder_name = folder_name[0].split("/")
        # ['.', 'data', 'cloudy']
        folder_name = folder_name[2]

        # 1. 폴더 구성
        folder_path = f"./dataset/{mode}/{folder_name}"
        os.makedirs(folder_path, exist_ok=True)

        # 2. 이미지 읽기
        img = cv2.imread(path)

        # 3. 이미지 저장
        # print(os.path.join(folder_path, image_name+".png"))
        # ./dataset/test/cloudy\train_28832.png
        cv2.imwrite(os.path.join(folder_path, image_name+".png"),img)



# 데이터 저장하기
all_train_list= []

all_val_list = []

all_test_list = []

for i in all_labels:

    all_train_list.append(globals()[i+"_train_list"])

    all_val_list.append(globals()[i+"_val_data"])

    all_test_list.append(globals()[i+"_test_data"])

# data_save(desert_test_data, mode="test")
for i in all_train_list:
    data_save(i, mode="train")

for i in all_val_list:
    data_save(i, mode="val")

for i in all_test_list:
    data_save(i, mode="test")