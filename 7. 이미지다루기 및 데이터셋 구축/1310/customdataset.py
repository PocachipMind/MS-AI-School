from torch.utils.data import Dataset
import os
import glob
from PIL import Image

class my_custom(Dataset) :
    def __init__(self, path, transform=None):
        self.all_path = glob.glob(os.path.join(path, "*", "*.jpg"))
        self.transform = transform
        self.label_dict = {"cats" : 0 , "dogs" : 1}

    def __getitem__(self, item):
        image_path = self.all_path[item]
        image = Image.open(image_path).convert("RGB")

        label_temp = image_path.split("\\")[1]
        label = self.label_dict[label_temp]

        if self.transform is not None :
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.all_path)