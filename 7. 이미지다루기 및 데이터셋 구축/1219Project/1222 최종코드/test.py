import torch
import cv2
from torchvision import models
from dataset_temp import custom_dataset
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


def acc_function(correct, total):
    acc = correct / total * 100
    return acc

def test(model, test_loader, device):

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad() :
        for batch_idx, (data, target, path) in enumerate(test_loader):
            image_path = path[0]
            img = cv2.imread(image_path)

            data, target = data.to(device), target.to(device)
            output = model(data)
            _, argmax = torch.max(output, 1)
            total += target.size(0)
            correct += (target == argmax).sum().item()

            argmax_temp = argmax.item()
            cv2.putText(img, str(argmax_temp), (90,90), cv2.FONT_ITALIC, 1, (0,0,0), 2)

            cv2.imshow("Image", img)
            cv2.waitKey(0)
        acc = acc_function(correct, total)
        print("acc for {} image : {:.2f}%".format(total, acc))

def main() :
    # val aug
    val_transform = A.Compose([
        A.Resize(height=224, width=224),
        ToTensorV2()
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = models.__dict__["resnet18"](pretrained=False, num_classes=4)
    net = net.to(device)

    net.load_state_dict(torch.load("./model_save/final.pt", map_location=device))
    test_data = custom_dataset("./data/val", transform=val_transform)
    test_loader = DataLoader(test_data,batch_size=1,shuffle=False)
    test(net, test_loader, device)

if __name__ == "__main__" :
    main()
