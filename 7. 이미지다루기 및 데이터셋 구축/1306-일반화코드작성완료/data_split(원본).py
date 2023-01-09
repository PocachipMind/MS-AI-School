import os
import glob
from sklearn.model_selection import train_test_split
import cv2

# 데이터 나눌 필요가 있음
# 학습, 중간평가, 테스트
# 학습 8 중간 평가 1 테스트 1 비율로 나눌것임
# 데이터 구조
"""
    -train
        -각 라벨별 폴더
            - image.jpg
    -val
        - 각 라벨별 폴더
    - test
        - 각 라벨별 폴더
"""

# 1. 이미지 폴더 가져오기
img_cloudy_path = "./data/cloudy"
img_cloudy = glob.glob(os.path.join(img_cloudy_path, "*.jpg"))

img_desert_path = "./data/desert"
img_desert = glob.glob(os.path.join(img_desert_path, "*.jpg"))

img_green_area_path = "./data/green_area"
img_green_area = glob.glob(os.path.join(img_green_area_path, "*.jpg"))

img_water_path = "./data/water"
img_water = glob.glob(os.path.join(img_water_path, "*.jpg"))

# print(len(img_cloudy), len(img_desert), len(img_green_area), len(img_water), )
# 1500 1131 1500 1500

# 2. train, val, test 나누기
cloudy_train_list, cloudy_val_list = train_test_split(img_cloudy, test_size=0.2, random_state=7)
cloudy_val_data, cloudy_test_data = train_test_split(cloudy_val_list, test_size=0.5, random_state=7)

desert_train_list, desert_val_list = train_test_split(img_desert, test_size=0.2, random_state=7)
desert_val_data, desert_test_data = train_test_split(desert_val_list, test_size=0.5, random_state=7)

green_area_train_list, green_area_val_list = train_test_split(img_green_area, test_size=0.2, random_state=7)
green_area_val_data, green_area_test_data = train_test_split(green_area_val_list, test_size=0.5, random_state=7)

water_train_list, water_val_list = train_test_split(img_water, test_size=0.2, random_state=7)
water_val_data, water_test_data = train_test_split(water_val_list, test_size=0.5, random_state=7)


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

all_train_list= [cloudy_train_list, desert_train_list, green_area_train_list, water_train_list]

all_val_list = [cloudy_val_data,desert_val_data, green_area_val_data,water_val_data]

all_test_list = [cloudy_test_data, desert_test_data, green_area_test_data, water_test_data ]


# data_save(desert_test_data, mode="test")
for i in all_train_list:
    data_save(i, mode="train")

for i in all_val_list:
    data_save(i, mode="val")

for i in all_test_list:
    data_save(i, mode="test")
