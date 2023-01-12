import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from customdataset import my_custom
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1 transforms
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.RandomEqualize(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])
val_transforms = transforms.Compose([
    transforms.Resize((224,244)),
    transforms.ToTensor()
])
# mean=[0.485, 0.456, 0.406] std=[0.229, 0.224, 0.225]
# 2 data set data loader
train_dataset = my_custom("./dataset/training_set/", transform=train_transforms)
val_dataset = my_custom("./dataset/test_set/", transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# 3 model call
net = models.__dict__["resnet18"](pretrained=False)
# print(net)
net.fc = torch.nn.Linear(in_features=512, out_features=2)
net.to(device)
# 4 train loop
def train(num_epoch, model, train_loader, val_loader, criterion, optimizer,
        device) :
    print("train ....!!! ")
    total = 0
    best_loss = 7777

    for epoch in range(num_epoch) :
        for i, (images, labels) in enumerate(train_loader) :
            img, labels = images.to(device), labels.to(device)

            # model <- img
            output = model(img)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, argmax = torch.max(output, 1)
            acc = (labels == argmax).float().mean()
            total += labels.size(0)

            if (i+1) % 10 == 0 :
                print("Epoch [{}/{}] Step [{}/{}] Loss {:.4f} Acc {:.2f}".format(
                    epoch + 1,
                    num_epoch,
                    i + 1,
                    len(train_loader),
                    loss.item(),
                    acc.item() * 100
                ))
        avrg_loss, val_acc = validation(epoch, model, val_loader, criterion, device)

        if avrg_loss < best_loss :
            print("Best acc save !!! ")
            best_loss = avrg_loss
            torch.save(model.state_dict(), "./best.pt")

    torch.save(model.state_dict(), "./last.pt")

# 5. val loop
def validation(epoch, model, val_loader, criterion, device) :
    print(f"validation .... {epoch} ")
    model.eval()
    with torch.no_grad() :
        total = 0
        correct = 0
        total_loss = 0
        cnt = 0
        batch_loss = 0
        for i, (images, labels) in enumerate(val_loader):
            img, label = images.to(device), labels.to(device)

            # model <- img
            output = model(img)
            loss = criterion(output, label)
            batch_loss += loss.item()

            total += img.size(0)
            _, argmax = torch.max(output, 1)
            correct += (label == argmax).sum().item()
            total_loss += loss
            cnt += 1

    avrg_loss = total_loss / cnt
    val_acc = (correct / total * 100)
    print("acc : {:.2f}% loss : {:.4f}".format(
        val_acc, avrg_loss
    ))
    model.train()
    return avrg_loss, val_acc

# 0 Hyper parameter
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)

def test(model, val_loader, device) :
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad() :
        for i, (image, labels) in enumerate(val_loader) :
            image, label = image.to(device), labels.to(device)
            output = model(image)
            _, argmax = torch.max(output, 1)

            total += image.size(0)
            print(image.size(0), total) # 128
            correct += (label == argmax).sum().item()

        acc = correct / total * 100
        print("acc for {} image : {:.2f}%".format(
            total, acc
        ))



if __name__ == "__main__" : # main
    net.load_state_dict(torch.load("./best.pt", map_location=device))
    test(net, val_loader, device)

# train(num_epoch=100, model=net, train_loader=train_loader, val_loader=val_loader,
    #       criterion=criterion, optimizer=optimizer, device=device)