import glob
import os
import cv2
import numpy as np
import torch
from config import TEST_DIR, CLASSES, DEVICE
from model import create_model

# model call
model = create_model(num_classes=5).to(DEVICE)
model.load_state_dict(torch.load("../outputs/model100.pth", map_location=DEVICE))
model.eval()

test_images = glob.glob(os.path.join(TEST_DIR, "*.jpg"))
print(f"Test instances : {len(test_images)}")

detection_threshold=0.85 # 점수 85이상인것만 보여주기

for i in range(len(test_images)) :
    image_name = test_images[i].split("\\")[-1].split('.')[0]

    # image read
    image = cv2.imread(test_images[i])
    orig_image = image.copy()

    # BRG to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32) # np.float_ 로 바꿔야 할 수 있음.

    # mask the pixel range between 0 and 1 정규화(노말제이션)
    image /= 255.0

    # bring color channels to front
    image = np.transpose(image, (2,0,1)).astype(np.float_)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda() # cpu는 cpu로
    # add batch dimension
    image = torch.unsqueeze(image,0) # 차원늘리기
    with torch.no_grad() :
        outputs = model(image)

    # load all detection to cpu for further operations ( cpu인 이유 : 쓰기 편하려고 )
    outputs = [{k : v.to('cpu') for k, v in t.items()} for t in outputs]

    if len(outputs[0]['boxes']) !=0 :
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to 'detection_threshold'
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes) :
            cv2.rectangle(orig_image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0,0,255), 2)

            cv2.putText(orig_image, pred_classes[j],
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),
                        2)
            cv2.putText(orig_image, str(scores[j]),
                        (int(box[0]), int(box[1]-30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),
                        2)

        cv2.imshow("Prediction", orig_image)
        cv2.waitKey(0)
        os.makedirs("../test_result/",exist_ok=True)
        cv2.imwrite(f"../test_result/{image_name}.png", orig_image)

    print(f"Image {i+1} done ...")
    print("-"*50)

print("TEST PREDICTIONS COMPLETE ...")


