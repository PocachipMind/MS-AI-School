import os
import glob
import cv2
from xml.etree.ElementTree import parse
xml_dir = "./wine labels_voc_dataset/"

class Voc_to_yolo_convter() :
    def __init__(self, xml_paths):
        self.xml_path_list = glob.glob(os.path.join(xml_paths,"*.xml"))

    def get_voc_to_yolo(self):
        for xml_path in self.xml_path_list :
            tree = parse(xml_path)
            root = tree.getroot()

            # get file name
            file_name = root.find('filename').text
            print(file_name)

            # get image size
            size_meta = root.findall('size')
            img_width = int(size_meta[0].find('width').text)
            img_height = int(size_meta[0].find('height').text)

            # object meta
            object_metas = root.findall('object')

            # box info get
            for object_meta in object_metas :
                # label_name
                object_label = object_meta.find('name').text

                # bbox
                xmin = int(object_meta.find('bndbox').findtext('xmin'))
                xmax = int(object_meta.find('bndbox').findtext('xmax'))
                ymin = int(object_meta.find('bndbox').findtext('ymin'))
                ymax = int(object_meta.find('bndbox').findtext('ymax'))

                # print(object_label, xmin, ymin, xmax, ymax)
                # voc to yolo
                yolo_x = round(((int(xmin) + int(xmax))/2)/img_width, 6)
                yolo_y = round(((int(ymin) + int(ymax))/2)/img_height,6)
                yolo_w = round((int(xmax) - int(xmin))/img_width, 6)
                yolo_h = round((int(ymax) - int(ymin))/img_height, 6)

                print(yolo_x, yolo_y, yolo_w, yolo_h)
        pass

if __name__ == "__main__" :
    test = Voc_to_yolo_convter(xml_dir)
    test.get_voc_to_yolo()