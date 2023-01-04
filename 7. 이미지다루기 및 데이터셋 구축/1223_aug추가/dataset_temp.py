import cv2
import glob
import os

from torch.utils.data import Dataset

label_dict = {"dekopon": 0,  "grapefruit": 1,  "kanpei": 2, "orange": 3}


class custom_dataset(Dataset):
    def __init__(self, image_file_path, transform=None):
        """
        data
            train
                라벨 폴더명 
                    이미지

        "./data/train"
        """
        self.image_file_paths = glob.glob(
            os.path.join(image_file_path, "*", "*.png"))
        self.transform = transform

    def __getitem__(self, index):
        # image loader
        image_path = self.image_file_paths[index]
        image = cv2.imread(image_path)

        # cv2 -> BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # label
        label_temp = os.path.basename(image_path)
        label_temp = label_temp.split("_")
        label_temp = label_temp[0]
        label = label_dict[label_temp]

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        image = image.float()
        return image, label

    def __len__(self):
        return len(self.image_file_paths)


# 디버깅
# if __name__ == '__main__':
#     test = custom_dataset("./data/train", transform=None)
#     for i in test:
#         pass
