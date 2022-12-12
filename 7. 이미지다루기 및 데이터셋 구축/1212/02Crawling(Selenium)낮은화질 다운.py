from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from multiprocessing import Pool
import time
import os
import urllib.request
import pandas as pd

# 폴더 생성
def create_folder(directory):
    try :
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('error : Creating directory ...' + directory)


# 검색 키워드 호출
# text라도 read_csv 상관없음
key = pd.read_csv('./keyword.txt', names=['keyword']) # 맥의 경우 encoding='' 필요
keyword = []
[keyword.append(key['keyword'][x]) for x in range(len(key))]
# print(keyword)


# 이미지 다운로드

def image_download(keywords):
    create_folder("./" + keywords + "_low_resolution")

    # 크롬 드라이브 호출
    options = webdriver.ChromeOptions()
    options.add_experimental_option("detach", True)
    chromedriver = "./chromedriver"
    driver = webdriver.Chrome(chromedriver, options=options)
    driver.implicitly_wait(3)

    # 검색
    print('검색 >> ', keywords)
    driver.get("https://www.google.co.kr/imghp?h1=ko")
    keyword = driver.find_element_by_xpath(
        '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input')
    keyword.send_keys(keywords)
    keyword.send_keys(Keys.RETURN)

    # 스크롤 내리기 -> 결과 더보기 버튼 클릭
    print("스크롤 ..... ", keywords)
    elem = driver.find_element_by_tag_name('body')
    for i in range(60):
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)

    try:
        # //*[@id="islmp"]/div/div/div/div[2]/div[1]/div[2]/div[2]/input
        driver.find_element_by_xpath(
            '//*[@id="islmp"]/div/div/div/div[2]/div[1]/div[2]/div[2]/input').click()
        for i in range(60):
            elem.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.2)
    except:
        pass

    links = []
    images = driver.find_elements_by_css_selector("img.rg_i.Q4LuWd")
    for image in images:
        if image.get_attribute('src') != None:
            links.append(image.get_attribute('src'))

    print(keywords + '찾은 이미지 개수 : ', + len(links))
    time.sleep(2)

    for index, i in enumerate(links):
        url = i
        start = time.time()
        urllib.request.urlretrieve(
            url, "./" + keywords + "_low_resolution/" + keywords + "_" + str(index) + ".jpg")
        print(str(index+1) + "/" + str(len(links)) + " " + keywords +
              " 다운로드 시간 ------ : ", str(time.time() - start)[:5] + '초')
    print(keywords,"다운로드 완료 !!!!")



# ===========================================================================
# 실행
# ===========================================================================
if __name__ == '__main__':
    pool = Pool(processes=3)
    pool.map(image_download, keyword)