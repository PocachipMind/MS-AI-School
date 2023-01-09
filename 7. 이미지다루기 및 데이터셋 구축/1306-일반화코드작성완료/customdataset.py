import os
import glob
from PIL import Image

from torch.utils.data import Dataset


class customDataset(Dataset) :
    def __init__(self, path , all_labels=os.listdir("./data") , transform=None):
        # path -> dataset/train/
        self.all_image_path = glob.glob(os.path.join(path,"*","*.png"))
        self.transform = transform
        self.label_dict = {}
        for index, labels in enumerate(all_labels):
            self.label_dict[labels] = index

        # print(self.label_dict)


    def __getitem__(self, item):
        img_path = self.all_image_path[item]
        # './dataset/train\\{라벨}\\train_10021.png'
        label_temp = img_path.split("\\")
        label = self.label_dict[label_temp[1]]
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.all_image_path)

if __name__ == "__main__":
    path = "./data"
    all_labels = os.listdir(path)

    test = customDataset("./dataset/train", all_labels , transform=None)
    for image,label in test :
        image.show()
        print(all_labels[label])