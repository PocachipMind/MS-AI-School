# train loop
# val loop
# 모델 save
# 평가 함수
import torch
import os
import torch.nn as nn # 모델에 대한 아키텍쳐 만들때 쓰임.
from metric_monitor_temp import MetricMonitor
from tqdm import tqdm

# 평가 함수
def calculate_acc(output, target):
    output = torch.sigmoid(output) >= 0.5
    target = target == 1.0
    return torch.true_divide((output == target).sum(dim=0), output.size(0).item())

# 모델 save
def save_model(model,save_dir, file_name='last.pt'):
    # save model
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, file_name)
    if isinstance(model, nn.DataParallel):
        print('멀티 GPU 저장 !! ')
        torch.save(model.module.state_dict(), output_path) # 모델의 상태 정보 저장
    else:
        print('싱글 GPU 저장 !! ')
        torch.save(model.state_dict(), output_path) # 모델의 상태 정보 저장

# train loop
def train(train_loader, model, criterion, optimizer, epoch, device):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for batch_idx, (image, target) in enumerate(train_loader):
        images = image.to(device)
        '''
        [ 0, 0, 0, 0
          0, 0, 0, 0
          0, 0, 0, 0].("cuda")
        images 는 이런식으로 나올것임.
        '''
        target = target.to(device)
        output = model(images) # 예측한 결과치가 output으로 나옴.
        loss = criterion(output, target) # 정답 맞췄는지 보는거
        accuracy = calculate_acc(output, target)

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy.item())
        optimizer.zero_grad() # 초기화
        loss.backward() # 로스값을 백워드해서
        optimizer.step() # 옵티마이저 단계설정

        stream.set_description(
            f"Epoch : {epoch}.    Train... {metric_monitor}".format(
                epoch=epoch, metric_monitor=metric_monitor
            )
        )

# val loop
def validate(val_loader, model, criterion, epoch, device):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    for batch_idx, (image, target) in enumerate(val_loader):
        images = image.to(device)
        target = target.to(device)
        output = model(images)
        loss = criterion(output, target)
        accuracy = calculate_acc(output, target)

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy.item())
        stream.set_description(
            f"Epoch : {epoch}.     Val... {metric_monitor}".format(
                epoch=epoch, metric_monitor=metric_monitor
            )
        )