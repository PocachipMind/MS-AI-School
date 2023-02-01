import os
import json
import cv2
import copy

def main():
    # 데이터셋 최상단 경로
    data_root = r'A:/test/park_data'

    for use in ['valid', 'train']:
        image_root = os.path.join(data_root, use, 'image')
        label_root = os.path.join(data_root, use, 'label')

        for image_path in get_image_paths(image_root):
            # A:/test/park_data\train\image\train_data\illegal\banner\13_dsp_su_10-27_16-29-04_aft_DF5.jpg
            # ['A:/test/park_data', 'valid', 'image', 'valid_data', 'illegal', 'banner', '13_dsp_su_10-27_16-48-57_aft_DF5.jpg']

            # image, json matching
            dirs = image_path.split('\\')
            label_path = os.path.join(label_root, dirs[3], dirs[4], dirs[5])

            filename = os.path.basename(image_path).split('.')[0] + '.json'
            label_path = os.path.join(label_path, filename)

            if not os.path.isfile(label_path):
                # print('File does not exist:', label_path)
                pass

            # print(image_path, label_path)
            anno_data = read_json(label_path)
            # image = visualize(image_path, anno_data)

            image, anno_data = resize(image_path, anno_data, (1470, 810))
            image = visualize_test(image, anno_data)

            cv2.imshow('visual', image)
            # if cv2.waitKey(0) & 0xff == ord('q'):
            #     cv2.destroyAllWindows()
            #     exit()

            # 이미지 확인 & 키보드 클릭 이벤트 처리: keyboard 버튼 클릭과 (if문 하위에)동작 매칭 - 저장, 삭제 등
            while True:
                key = cv2.waitKey()
                if key == ord('s'):
                    pass
                elif key == 27: # ESC
                    cv2.destroyAllWindows()
                    exit()
                else: # 위 if문에서 지정하지 않은 키보드 입력인 경우 다음 이미지로 넘어감
                    break

def visualize_test(image, anno_data):
    # resize 결과 시각화 용 코드
    for anno in anno_data['annos']:
        # rectangle
        pt1 = (anno['bbox'][0], anno['bbox'][1])
        pt2 = (anno['bbox'][2], anno['bbox'][3])
        image = cv2.rectangle(image, pt1, pt2, (250, 0, 250), 5)

        # text
        pt = (anno['bbox'][0], anno['bbox'][3] - 20)
        image = cv2.putText(image, anno['label'], pt, cv2.FONT_HERSHEY_SIMPLEX, 2, (250, 0, 250), 1, cv2.LINE_AA)

    image = cv2.resize(image, (960, 540))

    return image





def get_image_paths(root_path):
    # root_path 경로 하위에 있는 모든 파일 경로 탐색
    paths = []
    for (path, dir, files) in os.walk(root_path):
        for file in files:
            if file.split('.')[-1] != 'jpg':
                continue

            image_path = os.path.join(path, file)
            paths.append(image_path)
    return paths

def read_json(json_path):
    # 본 데이터셋에서 학습에 필요한 정보만 읽어 반환
    with open(json_path, 'r', encoding="utf8") as j:
        json_data = json.load(j)

    images = json_data['images']
    annotations = json_data['annotations']

    filename = images['ori_file_name']
    height = images['height']
    width = images['width']

    annos = []
    for annotation in annotations:
        label = annotation['object_class']
        bbox = annotation['bbox']
        xmin, ymin = bbox[0]
        xmax, ymax = bbox[1]

        anno = {
            'label': label,
            'bbox': [int(xmin), int(ymin), int(xmax), int(ymax)]
        }
        annos.append(anno)

    data = {
        'filename': filename,
        'height': height,
        'width': width,
        'annos': annos
    }

    return data

def visualize(image_path, anno_data):
    # label과 bbox 시각화
    image = cv2.imread(image_path)

    for anno in anno_data['annos']:
        # rectangle
        pt1 = (anno['bbox'][0], anno['bbox'][1])
        pt2 = (anno['bbox'][2], anno['bbox'][3])
        image = cv2.rectangle(image, pt1, pt2, (250, 0, 250), 5)

        # text
        pt = (anno['bbox'][0], anno['bbox'][3] - 20)
        image = cv2.putText(image, anno['label'], pt, cv2.FONT_HERSHEY_SIMPLEX, 2, (250, 0, 250), 1, cv2.LINE_AA)

    # 원본 이미지가 크므로 원활한 시각화를 위해 이미지 크기 조절
    image = cv2.resize(image, (960, 540))

    return image


def resize(image_path, anno_data, size):
    # size: (new_width, new_height)
    # size 크기로 맞춰 이미지 resize 및 bbox 정보 수정
    
    width = anno_data['width']
    height = anno_data['height']

    image = cv2.imread(image_path)
    image = cv2.resize(image, (size[0], size[1]))

    width_ratio = size[0] / width
    height_ratio = size[1] / height

    new_data = copy.deepcopy(anno_data)
    for anno in new_data['annos']:
        xmin, ymin = anno['bbox'][0], anno['bbox'][1]
        xmax, ymax = anno['bbox'][2], anno['bbox'][3]

        xmin, xmax = xmin * width_ratio, xmax * width_ratio
        ymin, ymax = ymin * height_ratio, ymax * height_ratio

        anno['bbox'] = [int(xmin), int(ymin), int(xmax), int(ymax)]


    return image, new_data


if __name__ == '__main__':
    main()