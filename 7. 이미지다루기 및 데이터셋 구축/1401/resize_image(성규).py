import os
import cv2

image_file_path = './dataset/valid/images/'
save_file = './dataset/valid/'
file_list = os.listdir(image_file_path)
for i in file_list:
    image_path = os.path.join(image_file_path, i)
    img = cv2.imread(image_path)
    img_re = cv2.resize(img,(960,540))

    
    image_resize_path = os.path.join(save_file,"resize", i)
    os.makedirs(os.path.join(save_file,"resize"), exist_ok=True)
    print(image_resize_path)
    cv2.imwrite(image_resize_path, img_re)
    
