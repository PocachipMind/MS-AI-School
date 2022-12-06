# Word Cloud : 단어 뭉치를 가시화하는 방식이다. 빈도수/중요도 등 가중치에 따라 색/크기로 차이를 주어 표현한다.

import pytagcloud
import pygame
import webbrowser
# pip install pytagcloud/ pygame/simplejson

# 가중치 정해주기
tag = [('hello', 100), ('world', 80), ('python', 120), ('kdb', 70),
       ('nice', 60), ('Deep Learning', 20), ('DB', 40),
       ('great', 120), ('MySQL', 110), ('DT', 125),
       ('SVM', 10), ('Text Data mining', 170), ('kaggle', 45),
       ('randomForest', 55), ('Regression', 160), ('Loss Function', 195)]

tag_list = pytagcloud.make_tags(tag, maxsize=50) # tag화 시켜줌
pytagcloud.create_tag_image(tag_list, 'word_cloud.jpg', size=(900, 600), rectangular=False)

webbrowser.open('word_cloud.jpg')