from PIL import Image
# pip install Pillow > 이 친구도 이미지 라이브러리임. cv2와 비슷함

def expand2square(pil_img, background_color=(0, 0, 0)):
    width, height = pil_img.size # 이미지 가져오기
    # print(width, height)
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        # image add (추가 이미지 , 붙일 위치 = (가로 , 세로 ))
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


img = Image.open('./images.png')
# print(img) # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=174x290 at 0x1BF4909BD30>
img_new = expand2square(img, (0, 0, 0)).resize((224, 224))
img_new.save("./test.png", quality=100)