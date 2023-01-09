import cv2
import torch
import torchvision.models as models
import torch.nn as nn
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
import numpy as np

labels = {
    0: 'paper',
    1: 'rock',
    2: 'scissors'
}

data_transforms = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.Resize(width=224, height=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
])

def processing(image):
    image = Image.fromarray(image)
    image = np.array(image)
    image = data_transforms(image=image)['image']
    # print(image.shape)
    image = image.unsqueeze(0)
    return image

def argmax(prediction):
    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()
    top_1 = np.argmax(prediction, axis=1)
    score = np.amax(prediction)
    score = '{:6f}'.format(score)
    prediction = top_1[0]
    result = labels[prediction]

    return result, score

###### 동영상 속성 확인 ######
cap = cv2.VideoCapture(0)  # 0을 넣으면 웹 캡을 가져오게 됩니다. ( 연결 되어있는 카메라 디바이스 가져오는거. )
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)

print('width :', width, 'height :', height)
print('fps :', fps)
print('frame_count: ', frame_count)

'''
width : 640.0 height : 480.0
fps : 30.0
frame_count:  -1.0
'''

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = models.shufflenet_v2_x2_0(pretrained=False)
model.fc = nn.Linear(in_features=2048, out_features=3)

model.load_state_dict(torch.load("best.pt", map_location=device))

model.eval()

###### 동영상 파일 읽기 ######
if cap.isOpened():  # 캡쳐 객체 초기화 확인
    while True:
        ret, frame = cap.read()
        image_data = processing(frame)
        output = model(image_data)
        result, score = argmax(output)

        str = result
        if not ret:  # 새로운 프레임 없다면 종료
            break
        cv2.putText(frame, str, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.imshow("video file show", frame)
        if cv2.waitKey(25) == 27:  # 25ms 기다리고 프레임 전환, esc 누르면 종료되도록 설정
            break
else:
    print('비디오 파일 읽기 실패')

cap.release()
cv2.destroyAllWindows()
