from config import DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR
from config import VISUALIZE_TRANSFORMED_IMAGES
from config import SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH
from model import create_model
from tqdm.auto import tqdm
from datasets import train_loader, valid_loader
from utils import Averager
import torch
import matplotlib.pyplot as plt
import time

plt.style.use('ggplot') # 그레이 격자 만들어주기

# function for running training iterations
def train(train_data_loader, model) :
    print("Training ....")
    # 훈련이 끝날 때마다 반복횟수와 손실값 저장
    global train_itr
    global train_loss_list

    # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

    for i, data in enumerate(prog_bar) :

        optimizer.zero_grad()
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k ,v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value) # 로스 업데이트

        losses.backward()
        optimizer.step()

        train_itr +=1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss : {loss_value:.4f}")

    return train_loss_list

def validate(valid_data_loader, model) :
    print("Validating")
    global val_itr
    global val_loss_list

    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    for i, data in enumerate(prog_bar) :

        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k ,v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)

        val_itr += 1

        prog_bar.set_description(desc=f"Loss : {loss_value:.4f}")

    return val_loss_list

# main code
if __name__ == "__main__" :
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]

    # define the optimizer
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    # initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1

    train_loss_list = []
    val_loss_list = []

    # name to save the trained model with
    MODEL_NAME = 'model'

    if VISUALIZE_TRANSFORMED_IMAGES :
        from utils import show_tranformed_image
        show_tranformed_image(train_loader)

    # start the training epochs
    for epoch in range(NUM_EPOCHS) :
        print(f"\nEpoch {epoch+1} of {NUM_EPOCHS}")

        train_loss_hist.rest()
        val_loss_hist.rest()

        # create two subplots one for each train and val
        figure_1, train_ax = plt.subplots()
        figure_2, val_ax = plt.subplots()

        # start timer and carry out training and validation
        start = time.time()
        train_loss = train(train_loader,model)
        val_loss = validate(valid_loader, model)
        print(f"Epoch # {epoch} train loss : {train_loss_hist.value:.3f}")
        print(f"Epoch # {epoch} validation loss : {val_loss_hist.value:.3f}")
        end = time.time()
        print(f"Took {((end - start)/60):.3f} minutes for epoch # {epoch}")

        if (epoch+1) % SAVE_MODEL_EPOCH == 0 : # save model after every n epochs
            torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch+1}.pth")

        if (epoch+1) % SAVE_PLOTS_EPOCH == 0 : # save loss plots after n epochs
            train_ax.plot(train_loss, color ='blue') # 그리기
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            val_ax.plot(val_loss, color='red')
            val_ax.set_xlabel('iterations')
            val_ax.set_ylabel('val loss')
            figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch+1}.png")
            figure_2.savefig(f"{OUT_DIR}/val_loss_{epoch+1}.png")
            print("SAVEING PLOTS COMPLETE.....")

        if (epoch+1) == NUM_EPOCHS :
            train_ax.plot(train_loss, color="blue")
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            val_ax.plot(val_loss, color='red')
            val_ax.set_xlabel('iterations')
            val_ax.set_ylabel('val loss')
            figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch + 1}.png")
            figure_2.savefig(f"{OUT_DIR}/val_loss_{epoch + 1}.png")

            torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch+1}.pth")

        plt.close('all')