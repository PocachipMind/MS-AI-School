import matplotlib.pyplot as plt
import numpy as np
import torch

def train(n_epochs, train_loader, val_loader, model, optimizer, criterion, device,
          save_path, lass_validation_loss =None):

    valid_loss_min = np.Inf
    train_loss_ls = []
    valid_loss_ls = []

    for epoch in range(1, n_epochs+1) :
        train_loss = 0.0
        valid_loss = 0.0
        trainLoss = 0.0
        train_batch = 0
        validLoss = 0.0
        valid_batch = 0
        # train the model
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader) :
            # move to GPU
            datas, targets = data.to(device), target.to(device)
            output = model(datas)

            loss = criterion(output, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update training loss
            train_loss = train_loss + ((1 /(batch_idx + 1)) * (loss.data - train_loss))
            trainLoss += train_loss.item()
            train_batch += 1

        # val model
        model.eval()
        for batch_idx, (data, target) in enumerate(val_loader) :
            datas, targets = data.to(device), target.to(device)
            output = model(datas)
            loss = criterion(output, targets)

            valid_loss = valid_loss + ((1 / (batch_idx +1)) * (loss.data - valid_loss))
            validLoss += valid_loss.item()
            valid_batch += 1

        print("Epoch {} \t Training Loss : {:.6f} \t Validation Loss : {:.6f}". format(
            epoch,
            train_loss,
            valid_loss
        ))

        train_loss_ls.append(trainLoss/train_batch)
        valid_loss_ls.append(validLoss/valid_batch)

        if valid_loss <= valid_loss_min :
            print("Validation loss decreased ({:.6f} --> {:.6f}.) Saving model...". format(
                valid_loss_min,
                valid_loss,
            ))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss

    # loss 그래프 표현
    plt.plot(train_loss_ls, "-o")
    plt.plot(valid_loss_ls, "-o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "val"])
    plt.title("Train vs Val Loss")
    plt.show()
