# image data get -> train 80 val 10 test 10

# image -> cats get
# image -> dogs get

import cv2
import os
import glob
from sklearn.model_selection import train_test_split
import natsort
# pip install natsort >> 문자랑 숫자랑 섞인경우 솔트할때 얘 만한게 없음

cat_image_path = './image/cats/'
dog_image_path = './image/dogs/'

# cat 4000장
cat_image_full_path = glob.glob(os.path.join(f'{cat_image_path}/*.jpg'))
# print(cat_image_full_path) path 잇음
cat_image_full_path = natsort.natsorted(cat_image_full_path)
'''
cat_image_full_path = natsort.natsorted(
    glob.glob(os.path.join(f"{cat_image_path}/*.jpg")))
print(cat_image_full_path)
'''
print('cat image size >>', len(cat_image_full_path))

# dog 4000장
dog_image_full_path = natsort.natsorted(
    glob.glob(os.path.join(f"{dog_image_path}/*.jpg")))
print('dog image size >>', len(dog_image_full_path))

# train 80 val 20 
# val 20 --> val 10, test 10

# cat 데이터 나누기
cat_train_data, cat_val_data = train_test_split(cat_image_full_path, test_size=0.2, random_state=7777)
cat_val , cat_test = train_test_split(cat_val_data, test_size=0.5, random_state=7777)

print('cat train data :', len(cat_train_data), ', cat val data :', len(cat_val), ', cat test data :', len(cat_test))
# cat train data : 3200 , cat val data : 400 , cat test data : 400


# dog 데이터 나누기
dog_train_data, dog_val_data = train_test_split(dog_image_full_path, test_size=0.2, random_state=7777)
dog_val , dog_test = train_test_split(dog_val_data, test_size=0.5, random_state=7777)

print('dog train data :', len(dog_train_data), ', dog val data :', len(dog_val), ', dog test data :', len(dog_test))
# dog train data : 3200 , dog val data : 400 , dog test data : 400




# image cv2 imread -> 저장 하는 방법
# mv copy

# cat image data save
for cat_train_data_path in cat_train_data:
    # print(cat_train_data_path)
    img = cv2.imread(cat_train_data_path)
    os.makedirs('./dataset/train/cat/', exist_ok=True)
    file_name = os.path.basename(cat_train_data_path)
    cv2.imwrite(f'./dataset/train/cat/{file_name}', img)

# 개수가 같으므로 집으로 한번에 가능할듯!
for cat_val_path, cat_test_path in zip(cat_val, cat_test):
    img_val = cv2.imread(cat_val_path)
    img_test = cv2.imread(cat_test_path)
    file_name_val = os.path.basename(cat_val_path)
    file_name_test = os.path.basename(cat_test_path)
    os.makedirs('./dataset/val/cat/', exist_ok=True)
    os.makedirs('./dataset/test/cat/', exist_ok=True)
    cv2.imwrite(f'./dataset/val/cat/{file_name_val}', img_val)
    cv2.imwrite(f'./dataset/test/cat/{file_name_test}', img_test)



# dog image data save
for dog_train_path in dog_train_data:
    dog_train_img = cv2.imread(dog_train_path)
    dog_train_file_name = os.path.basename(dog_train_path)
    os.makedirs("./dataset/train/dog/", exist_ok=True)
    cv2.imwrite(f"./dataset/train/dog/{dog_train_file_name}", dog_train_img)

for dog_val_path, dog_test_path in zip(dog_val, dog_test):
    dog_val_img = cv2.imread(dog_val_path)
    dog_test_img = cv2.imread(dog_test_path)
    dog_val_name = os.path.basename(dog_val_path)
    dog_test_name = os.path.basename(dog_test_path)
    os.makedirs("./dataset/val/dog/", exist_ok=True)
    os.makedirs("./dataset/test/dog/", exist_ok=True)
    cv2.imwrite(f"./dataset/val/dog/{dog_val_name}", dog_val_img)
    cv2.imwrite(f"./dataset/test/dog/{dog_test_name}", dog_test_img)