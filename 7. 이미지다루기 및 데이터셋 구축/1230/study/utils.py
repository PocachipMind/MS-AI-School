import glob
import os
import torch.nn as nn
import torch
import torch.optim
import torchvision.models as models
import cv2

# from main2 import *

"""
def visualize_aug(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([
        t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))  # 정규화와 텐서화를 풀어줌
    ])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12,6))

    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()

    plt.tight_layout()
    plt.show()

# visualize_aug(train_dataset)
"""

def train(num_epoch, model, train_loader, val_loader, criterion,
          optimizer, scheduler, save_dir, device):
    print("Start Training ....")
    running_loss = 0.0
    total =0
    best_loss = 999
    for epoch in range(num_epoch+1):
        for i, (imgs, labels) in enumerate(train_loader):
            img, label = imgs.to(device), labels.to(device)
            # print(img)
            output = model(img)
            loss = criterion(output, label)
            scheduler.step()  # 스케쥴러 자리는 로스 밑에
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, argmax = torch.max(output, 1)
            acc = (label==argmax).float().mean()
            total += label.size(0)

            if (i + 1) % 10 == 0 :
                print("Epoch [{}/{}] Step[{}/{}] Loss :{:.4f} Acc : {:.2f}%".format(
                    epoch + 1, num_epoch, i+1, len(train_loader), loss.item(), acc.item()*100)
                )

        avrg_loss, val_acc = validation(model, val_loader, criterion, device)

        # 10 에포크 마다 모델 저장
        # if epoch % 10 == 0:
        #     save_model(model, save_dir, file_name=f"{epoch},pt")

        if avrg_loss < best_loss:
            print(f"Best save at epoch >> {epoch}")
            print("save model in ", save_dir)
            best_loss = avrg_loss
            save_model(model, save_dir)

    save_model(model, save_dir, file_name="last_resnet.pt")


def validation(model, val_loader, criterion, device):
    print("Start val")

    with torch.no_grad():
        model.eval()

        total = 0
        correct = 0
        total_loss = 0
        cnt = 0
        batch_loss = 0

        for i, (imgs, labels) in enumerate(val_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            loss = criterion(output, labels)
            batch_loss += loss.item()

            total += imgs.size(0)
            _,argmax = torch.max(output, 1)
            correct += (labels==argmax).sum().item()
            total_loss += loss
            cnt += 1

    avrg_loss = total_loss / cnt
    val_acc = (correct / total * 100)
    print("val Acc : {:.2f}% avg_loss : {:.4f}".format(
        val_acc, avrg_loss
    ))

    model.train()

    return avrg_loss, val_acc

def test_show(test_loader, device) :
    net = models.__dict__['resnet50'](pretrained=False)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 53)
    net.to(device)

    # train model call
    model_path = './weight/best_resnet.pt'
    net.load_state_dict(torch.load(model_path))

    test_data_path = './archive/test'
    label_dict = folder_name_det(test_data_path)
    # print(label_dict)

    net.eval()
    with torch.no_grad():
        for i, (imgs, labels, path) in enumerate(test_loader):
            inputs, outputs, paths = imgs.to(device), labels.to(device), path
            # print(paths)
            #('./archive/train\\ace of clubs\\001.jpg',) 튜플형태라서 그냥은 못읽음
            img = cv2.imread(paths[0])

            predicted_outputs = net(inputs)
            # print(predicted_outputs)
            _, predicted = torch.max(predicted_outputs, 1)
            # print("predicted >>", predicted)

            labels_temp = labels.item()

            labels_pr_temp = predicted.item()

            predicted_label = label_dict[str(labels_pr_temp)]
            answer_label = label_dict[str(labels_temp)]
            # print('answer_label >>', answer_label)
            # print("predicet_label >>", predicted_label)
            cv2.putText(img, predicted_label, (60, 180), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0))
            cv2.putText(img, answer_label, (60, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255))

            cv2.imshow("test", img)
            cv2.waitKey(0)

# 싱글 GPU일 경우 저장하는 함수
def save_model(model, save_dir, file_name="best_resnet.pt"):
    output_path = os.path.join(save_dir, file_name)

    torch.save(model.state_dict(), output_path)

def test_species(test_loader, device):
    net = models.__dict__['resnet50'](pretrained=False)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 53)
    net.to(device)

    # train model call
    model_path = './weight/best_resnet.pt'
    net.load_state_dict(torch.load(model_path))

    # folder_name_det() <- test_data_path
    test_data_path = './archive/test'
    label_det = folder_name_det(test_data_path)
    label_length = len(label_det)
    # list to calculate correct labels
    # print(label_length)  # 53
    labels_correct = list(0. for i in range(label_length))
    # list to keep the total
    labels_total = list(0. for i in range(label_length))

    total = 0
    correct = 0
    net.eval()

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_loader):
            inputs, outputs = imgs.to(device), labels.to(device)
            predicted_outputs = net(inputs)
            # 0.3 ,0.2, 0.5
            _, predicted = torch.max(predicted_outputs, 1)  # 0.5 반환

            labels_corrcet_running = (predicted == outputs).squeeze()
            # print(outputs)
            label = outputs[0]
            # print(label)
            # print(labels_correct)
            # print(labels_corrcet_running.item())
            '''
            True
            tensor([35], device='cuda:0')
            tensor(35, device='cuda:0')
            [120.0, 128.0, 168.0, 180.0, 128.0, 152.0, 144.0, 129.0, 149.0, 135.0, 131.0, 154.0, 154.0, 111.0, 150.0,
             134.0, 169.0, 159.0, 163.0, 170.0, 113.0, 126.0, 132.0, 122.0, 149.0, 123.0, 127.0, 128.0, 150.0, 155.0,
             156.0, 138.0, 157.0, 107.0, 120.0, 69.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0]
             '''
            if labels_corrcet_running.item():
                labels_correct[label] += 1
            labels_total[label] += 1
            total += inputs.size(0)
            correct += (outputs == predicted).sum().item()

        acc = correct / total * 100

        label_list = list(label_det.values())

        for i in range(53):
            print("Accuracy to predict %5s : %2d %%"% (label_list[i], 100*labels_correct[i] / labels_total[i]))

        print(f"Accuracy : {round(acc, 2)}%")

def folder_name_det(folder_path):
    # foler_path = ./dataset/test  folder_name = "*"
    folder_name = glob.glob(os.path.join(folder_path, "*"))
    # print(folder_name)
    # './archive/test\\ace of clubs'

    det = {}
    # 리스트 형태라서 스플릿 해야함
    for index, (path) in enumerate(folder_name):
    # for path in folder_name:
        temp_name = path.split('\\')
        # print(temp_name)
        # ['./archive/test', 'ten of spades']
        temp_name = temp_name[1]
        # print(temp_name, index)
        det[str(index)] = temp_name
    # print(det)
    return det

# folder_name_det('./archive/test')

'''
# 모델 input features 갯수 확인
net = models.__dict__['resnet50'](pretrained=False)
print(net)
# (fc): Linear(in_features=2048, out_features=1000, bias=True)
'''

