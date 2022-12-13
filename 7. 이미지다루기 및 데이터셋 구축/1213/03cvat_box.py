import os
import glob
import cv2
from xml.etree.ElementTree import parse

# xml 파일 찾을 수 있는 함수 제작
def find_xml_file(xml_folder_path):
    all_root = []
    for (path, dir, files) in os.walk(xml_folder_path):
        # print('...', path, dir, files)
        for filename in files :
            # image.xml -> .xml
            ext = os.path.splitext(filename)[-1]
            if ext == '.xml':
                root = os.path.join(path, filename)
                # ./cavt_annotations/annotations.xml
                all_root.append(root)
            else :
                print('No xml file...')
                continue
    return all_root

xml_dirs = find_xml_file('./cvat_annotations')
print(xml_dirs)

for xml_dir in xml_dirs:
    tree = parse(xml_dir)
    root = tree.getroot()
    img_metas = root.findall('image')
    for img_meta in img_metas:
        # xml 에 기록된 이미지 이름
        image_name = img_meta.attrib['name']
        print(image_name)

        image_path = os.path.join('./images', image_name)

        # image read
        image = cv2.imread(image_path)

        # image size info
        img_width = int(img_meta.attrib['width'])
        img_height = int(img_meta.attrib['height'])

        print(img_width , img_height)

        # box meta info
        box_metas = img_meta.findall('box')

        for box_meta in box_metas :
            box_label = box_meta.attrib['label']
            box = [int(float(box_meta.attrib['xtl'])), 
                   int(float(box_meta.attrib['ytl'])),
                   int(float(box_meta.attrib['xbr'])),
                   int(float(box_meta.attrib['ybr']))]
            print(box[0], box[1], box[2], box[3])

            rect_img = cv2.rectangle(
                image, (box[0],box[1]), (box[2],box[3]), (0, 255, 255), 2)
        
        # cv2.namedWindow('test')
        # cv2.moveWindow('test',300,300) # 윈도우 뜨는 거 위치 조정
        cv2.imshow("test", rect_img)
        cv2.waitKey(0)

