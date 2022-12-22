from torch.utils.data import Dataset
from xml.etree.ElementTree import parse


def box_xyxy(image_metas):
    list_box = []
    for img_meta in image_metas:
        box_metas = img_meta.findall('box')
        for box_meta in box_metas:
            box_label = box_meta.attrib['label']
            box = [int(float(box_meta.attrib['xt1'])),
                   int(float(box_meta.attrib['yt1'])),
                   int(float(box_meta.attrib['xbr'])),
                   int(float(box_meta.attrib['ybr']))]
            list_box.append(box)
    return list_box

class CustomDataset(Dataset):
    def __init__(self, dataset_path, xml_path):
        self.dataset_path = dataset_path
        self.xml_path = xml_path
    
    def __getitem__(self, index):
        image_path = self.dataset_path[index]
        xml_path = self.xml_path[index]
        tree= parse(self.xml_path)
        root = tree.getroot()
        image_metas = root.findall('image')
        box = box_xyxy(image_metas)

        # 리턴에 정답은 없습니다.
        # return image, box, label
    
    def __len__(self):
        return len(self.dataset_path)


image_path = ["./01.png"]
xml_path = ['./annotations.xml']
test = CustomDataset(image_path, xml_path)

for i in test:
    pass