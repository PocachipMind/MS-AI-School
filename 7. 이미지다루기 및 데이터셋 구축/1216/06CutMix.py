import os
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

image_path = './cutMix_data'
index_len = len(os.listdir(image_path))
print('index_len ', index_len)
image_list = os.listdir(image_path)
print(' image list >> ', image_list)


# loader image
def load_image(path, index):
    image = cv2.imread(os.path.join(path, image_list[index]), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0 # 이미지 데이터 값 정규화

    return image



image = load_image(image_path, 3)
image_size = image.shape[0]

def cutmix(path, index, imsize):
    w, h = imsize, imsize
    scale_s = imsize, imsize
    s = imsize // 2

    # 중앙값 랜덤하게 잡기
    # x center, y center
    xc, yc = [int(random.uniform(imsize*0.25, imsize*0.75)) for _ in range(2)]
    # 256 ~ 768

    indexes = [index] + [random.randint(0, index) for _ in range(3)]

    # 검은색 배경의 임의 이미지 생성 ( 여기다가 이미지 붙여 넣기 )
    return_img = np.full((imsize, imsize, 3 ), 1, dtype=np.float32)

    for i, index in enumerate(indexes):
        image = load_image(path, index)

        # top left
        if i == 0:
            # large image
            x1a, y1a, x2a, y2a = max(xc-w, 0), max(yc-h, 0), xc, yc
            # small image
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:
            # top right
            x1a, y1a, x2a, y2a = xc, max(yc-h, 0), min(xc+w, s*2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:
            # bottom left
            x1a, y1a, x2a, y2a = max(xc-w, 0), yc, xc, min(s*2, yc+h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc , w), min(y2a - y1a, h)
        elif i == 3:
            # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s*2), min(s*2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        return_img[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
    
    return return_img


test = cutmix(image_path, 3, image_size)
plt.imshow(test)
plt.show()




