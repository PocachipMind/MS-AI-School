import torch
import cv2
import numpy as np
import os
import glob

from xml.etree import ElementTree as et
from config import CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE, NUM_SAMPLES_TO_VISUALIZE
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn, get_train_transform, get_valid_transform

# the dataset class
class MicrocontrollerDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.dir_path = dir_path
        self.width = width
        self.height = height
        self.classes = classes
        self.transforms = transforms
        # get all the image paths in sorted order
        self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
        self.all_images = [image_path.split(
            '/')[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # cpature the image name and the full image path
        image_name = self.all_images[idx]
        image_name = os.path.basename(image_name)
        image_path = os.path.join(self.dir_path, image_name)

        # read the image
        image = cv2.imread(image_path)
        # conver BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # resize for the required size
        image_resized = cv2.resize(image, (self.width, self.height))
        # normalize image
        image_resized /= 255.0

        # capture the corresponding xml file for getting the annotations
        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]

        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))

            # xmin = left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax
            ymax = int(member.find('bndbox').find('ymax').text)

            # resize the bounding boxes according to the ...
            xmin_final = (xmin / image_width)*self.width
            xmax_final = (xmax / image_width)*self.width
            ymin_final = (ymin / image_height)*self.height
            ymax_final = (ymax / image_height)*self.height

            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # apply the image transforms
        if self.transforms :
            sample = self.transforms(image = image_resized,
                                    bboxes = target['boxes'],
                                    labels=labels)


            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return image_resized,target

    def __len__(self):
        return len(self.all_images)


# prepare the final datasets and data loaders
# dataset
train_dataset = MicrocontrollerDataset(TRAIN_DIR, RESIZE_TO,
                                       RESIZE_TO, CLASSES,
                                       get_train_transform())
valid_dataset = MicrocontrollerDataset(VALID_DIR, RESIZE_TO,
                                       RESIZE_TO, CLASSES,
                                       get_valid_transform())

# dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers=2,
    collate_fn=collate_fn
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size = BATCH_SIZE,
    shuffle = False,
    num_workers = 2,
    collate_fn = collate_fn
)
# print(f"Number of training samples : {len(train_dataset)}")
# print(f"Number of validation samples : {len(valid_dataset)}")

if __name__ == "__main__" :
    dataset = MicrocontrollerDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES
    )
    print(f"Number of training images : {len(dataset)}")

    # function to visualize a single sample
    def visualize_sample(image, target) :
        box = target['boxes'][0]
        label = CLASSES[target['labels']]


        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            (0,255,0),
            2
        )

        cv2.putText(
            image,
            label,
            (int(box[0]), int(box[1]-5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,0,255),
            2
        )

        cv2.imshow("Image", image)
        cv2.waitKey(0)

    for i in range(NUM_SAMPLES_TO_VISUALIZE) :
        image, target = dataset[i]
        visualize_sample(image,target)