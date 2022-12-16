# pip install requests
import requests
import numpy as np
import io
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# url 다운 받고 그걸 array로 반환
def get_image_from_url(url):
    response = requests.get(url)
    img_pil = Image.open(io.BytesIO(response.content))

    return np.array(img_pil)

def mixup(x1, x2, y1, y2, lambda_=0.8):
    x = lambda_ * x1 + (1-lambda_) * x2
    y = lambda_ * y1 + (1-lambda_) * y2
    return x, y

# image url
cat_url = 'http://s10.favim.com/orig/160416/cute-cat-sleep-omg-Favim.com-4216420.jpeg'
dog_url = 'http://s7.favim.com/orig/150714/chien-cute-dog-golden-retriever-Favim.com-2956014.jpg'

cat_img = get_image_from_url(cat_url)
dog_img = get_image_from_url(dog_url)

x, y = mixup(cat_img, dog_img, np.array([1, 0]), np.array([0, 1]))
plt.axis("off")
plt.imshow(x.astype(int)), y
plt.show()












