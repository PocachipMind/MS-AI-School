import torch
from torchvision.transforms import transforms
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
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, argmax = torch.max(output, 1)
            total += target.size(0)
            correct += (target == argmax).sum().item()

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
