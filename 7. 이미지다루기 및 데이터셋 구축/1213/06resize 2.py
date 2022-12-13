from PIL import Image
import os


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    # print(width, height)
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        # image add (추가 이미지 , 붙일 위치 (가로 , 세로 ))
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def image_file(image_folder_path):
    all_root = []
    for (path, dir, files) in os.walk(image_folder_path):
        for filename in files:
            # image.xml -> .xml
            ext = os.path.splitext(filename)[-1]
            # ext_list = [".jpg", ".png", ".jpeg"]
            if ext == ".jpg":
                root = os.path.join(path, filename)
                # ./cavt_annotations/annotations.xml
                all_root.append(root)
            else:
                print("no image file..")
                continue
    return all_root

img_path_list = image_file('./images/')

for img_path in img_path_list:
    # image_name_temp = img_path.split('/') # 윈도우의 경우 \\ 필요
    image_name_temp = os.path.basename(img_path) # base네임 갖고오기
    image_name_temp2 = os.path.abspath(img_path) # 절대경로
    image_name = image_name_temp.replace(".png","")
    # print(image_name_temp) # >> kiwi_1

    img = Image.open(img_path)
    img_new = expand2square(img, (0, 0, 0)).resize((224, 224))
    os.makedirs("./resize", exist_ok=True)
    img_new.save(f"./resize/{image_name}.png", quality=100)