import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import cv2
from PIL import Image
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms

data_transforms = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)
labels = {0: "rock", 1: "scissors", 2: "paper"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = models.vgg19_bn(pretrained=False)
# model.classifier[6] = nn.Linear(in_features=4096, out_features=3)
model = models.mobilenet_v3_large(pretrained=False)
model.classifier[3] = nn.Linear(in_features=1280, out_features=3)

model.load_state_dict(torch.load("mb_best.pt", map_location=device))
model = model.to(device)
model.eval()
with torch.no_grad():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    while True:
        ret, frame = cap.read()
        # 왜 RGB 변환을 2번 해야 되는걸까?
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)
                                    # reshape() 보다 언스퀴즈 하는게 코드 수정이 덜함
        input_img = data_transforms(pil_img).unsqueeze(0).to(device)
        out = model(input_img)
        softmax_result = F.softmax(out)
        top1_prob, top1_label = torch.topk(softmax_result, 1)
        print(top1_prob, labels.get(int(top1_label)))
        acc = ">>  " + str(round(top1_prob.item()*100, 3)) + "%"
        cv2.putText(frame, labels.get(int(top1_label)), (10, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255, 0), 2)
        cv2.putText(frame, acc, (30, 160), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("TEST", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# real_time_classification()