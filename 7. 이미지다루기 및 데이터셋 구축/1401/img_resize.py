import os
import glob
import cv2

path = 'D:dataset/'

train_image_path = glob.glob(os.path.join(path, 'train', '*', '*.jpg'))
valid_image_path = glob.glob(os.path.join(path, 'valid', '*', '*.jpg'))

os.makedirs(path + 'resized_train', exist_ok=True)
os.makedirs(path + 'resized_valid', exist_ok=True)

for img_path in train_image_path:
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (960, 540))
    cv2.imwrite(path + 'resized_train/' + img_path.split('\\')[-1], img_resized)
    print(img_path)

for img_path in valid_image_path:
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (960, 540))
    cv2.imwrite(path + 'resized_valid/' + img_path.split('\\')[-1], img_resized)
    print(img_path)
