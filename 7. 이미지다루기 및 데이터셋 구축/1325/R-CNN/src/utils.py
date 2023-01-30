import albumentations as A  # pip install albumentations==1.0.3 했습니다. ( yolov5 권장 버전 )
import cv2
import numpy as np

from albumentations.pytorch import ToTensorV2
from config import DEVICE, CLASSES as classes

# 로스 업데이트 하기 위한 클래스
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    # 읽기 전용 필드 ( 붙어있는 함수만 )
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def rest(self):
        self.current_total = 0.0
        self.iterations = 0.0

# 서로 다른 영상이 서로 다른 개체 수를 가질 수 있으므로 데이터 로드를 처리하고 다양한 크기의 텐서도 처리합니다.
def collate_fn(batch):
    """
     To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    :param batch:
    :return: tuple(zip(*batch))
    """
    return tuple(zip(*batch))



# Training and Validation Augmentations
def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        A.RandomRotate90(0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        ToTensorV2()
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def get_valid_transform():
    return A.Compose([
        ToTensorV2()
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def show_tranformed_image(train_loder):
    if len(train_loder) > 0:
        for i in range(1):
            images, targets = next(iter(train_loder))
            images = list(image.to(DEVICE) for image in images)

            targets = [{k: v.to(DEVICE) for k, v in t.items()}
                       for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            for box in boxes:
                cv2.rectangle(sample,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 0, 255), 2)
            cv2.imshow("Transformed images", sample)
            cv2.waitKey(0)
