import os
import glob
import argparse
from PIL import Image
# image_pretreatment.py
# 오렌지 : Orange
# 자몽 : grapefruit
# 레드향 : Kanpei
# 한라봉 : Dekopon

# 폴더 구성 /dataset/image/각폴더명 생성/ 이미지 저장해주시면됩니다.(resize 400 x 400)
# 직사각형 -> 정사각형 리사이즈 비율 유지 하는 함수


def expand2square(img, background_color):

    width_temp, height_temp = img.size
    print(width_temp, height_temp)
    if width_temp == height_temp:
        return img
    elif width_temp > height_temp:
        result = Image.new(
            img.mode, (width_temp, width_temp), background_color)
        result.paste(img, (0, (width_temp - height_temp) // 2))
        return result
    elif width_temp < height_temp:
        result = Image.new(
            img.mode, (height_temp, height_temp), background_color)
        result.paste(img, ((height_temp - width_temp) // 2,  0))
        return result


def image_prossing(ornage_data, grapefruit_data, kanpei_data, dekopon_data):
    ornage = ornage_data
    grapefruit = grapefruit_data
    kanpei = kanpei_data
    dekopon = dekopon_data

    for i in dekopon:
        file_name = i.split('/')
        file_name_temp = file_name
        # file_name = i.split('\\')
        file_name = file_name[3]
        file_name = file_name.replace('.jpg', '.png')
        # file_name -> 77.jpg
        # ./image/orange/90.jpg
        dekopon_img = Image.open(i)
        dekopon_img_resize = expand2square(
            dekopon_img, (0, 0, 0)).resize((400, 400))

        # 폴더 생성
        os.makedirs("./dataset/image/dekopon", exist_ok=True)
        dekopon_img_resize.save(
            f"./dataset/image/dekopon/dekopon_{file_name}.png")

    for i in kanpei:
        file_name = i.split('/')
        file_name_temp = file_name
        # file_name = i.split('\\')
        file_name = file_name[3]
        file_name = file_name.replace('.jpg', '.png')
        # file_name -> 77.jpg
        # ./image/orange/90.jpg
        kanpei_img = Image.open(i)
        kanpei_img_resize = expand2square(
            kanpei_img, (0, 0, 0)).resize((400, 400))

        # 폴더 생성
        os.makedirs("./dataset/image/kanpei", exist_ok=True)
        kanpei_img_resize.save(
            f"./dataset/image/kanpei/kanpei_{file_name}")

    for i in grapefruit:
        file_name = i.split('/')
        file_name_temp = file_name
        # file_name = i.split('\\')
        file_name = file_name[3]
        file_name = file_name.replace('.jpg', '.png')
        # file_name -> 77.jpg
        # ./image/orange/90.jpg
        grapefruit_img = Image.open(i)
        grapefruit_img_resize = expand2square(
            grapefruit_img, (0, 0, 0)).resize((400, 400))

        # 폴더 생성
        os.makedirs("./dataset/image/grapefruit", exist_ok=True)
        grapefruit_img_resize.save(
            f"./dataset/image/grapefruit/grapefruit_{file_name}")

    for i in ornage:
        file_name = i.split('/')
        file_name_temp = file_name
        # file_name = i.split('\\')
        file_name = file_name[3]
        file_name = file_name.replace('.jpg', '.png')
        # file_name -> 77.jpg
        # ./image/orange/90.jpg
        orange_img = Image.open(i)
        print(file_name_temp, orange_img)
        orange_img_resize = expand2square(
            orange_img, (0, 0, 0)).resize((400, 400))

        # 폴더 생성
        os.makedirs("./dataset/image/orange", exist_ok=True)
        orange_img_resize.save(f"./dataset/image/orange/orange_{file_name}")


def image_file_check(opt):
    # image-folder-path
    # image_folder_path
    image_path = opt.image_folder_path
    # 각 폴더별 데이터 양 체크 #
    """
    image 
        - 자몽
            - xxx.jpg
        - 레드향
        - 한라봉 
    image_path -> ./image/orange/*.jpg
    """
    all_data = glob.glob(os.path.join(image_path, "*", "*.jpg"))
    print("전체 데이터 갯수 : ", len(all_data))
    # 오렌지
    ornage_data = glob.glob(os.path.join(image_path, "orange", "*.jpg"))
    print("오렌지 데이터 갯수 >> ", len(ornage_data))
    # 자몽
    grapefruit_data = glob.glob(os.path.join(
        image_path, "grapefruit", "*.jpg"))
    print("자몽 데이터 갯수 >> ", len(grapefruit_data))
    # 레드향
    kanpei_data = glob.glob(os.path.join(image_path, "kanpei", "*.jpg"))
    print("레드향 데이터 갯수 >> ", len(kanpei_data))
    # 한라봉
    dekopon_data = glob.glob(os.path.join(image_path, "dekopon", "*.jpg"))
    print("한라봉 데이터 갯수 >> ", len(dekopon_data))

    return ornage_data, grapefruit_data, kanpei_data, dekopon_data


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder-path", type=str, default="./image")
    opt = parser.parse_args()

    return opt


if __name__ == "__main__":
    opt = parse_opt()
    ornage_data, grapefruit_data, kanpei_data, dekopon_data = image_file_check(
        opt)
    image_prossing(ornage_data, grapefruit_data, kanpei_data, dekopon_data)
