# pip install selenium==4.2.0
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os
import urllib.request

#######
# 폴더 구성
#######
def create_folder(directory):
    try :
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('error : Creating directory ...' + directory)

# 크롬이 뜨자마자 꺼지는거 안되도록 설정
options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True)

# ===========
# 키워드 입력 , chromedriver 실행
# ===========

keywords = '사과'
# 윈도우 이므로 exe파일임
chromedriver_path = './chromedriver.exe'
driver = webdriver.Chrome(chromedriver_path, options=options)
driver.implicitly_wait(3)

#########
# 키워드 입력 selenum
########

driver.get('https://www.google.co.kr/imghp?h1=ko')

# find_element_by_name 이용하는 방법
# <input class="gLFyf" jsaction="paste:puy29d;" maxlength="2048" name="q" type="text" aria-autocomplete="both" aria-haspopup="false" autocapitalize="off" autocomplete="off" autocorrect="off" autofocus="" role="combobox" spellcheck="false" title="검색" value="" aria-label="검색" data-ved="0ahUKEwitrLDWgPP7AhWLwpQKHZ9BAJoQ39UDCAM">

elem = driver.find_element_by_name('q')
elem.send_keys(keywords)
elem.send_keys(Keys.RETURN)


# X Path로 쓸때
# input -> /html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input
# button -> /html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/button
# Keys.Return으로 엔터 눌러도됩니다.
'''
keyword = driver.find_element_by_xpath(
    '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input')
keyword.send_keys(keywords)
driver.find_element_by_xpath(
    '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/button').click()
'''

'''
크롬 킬 때 창 크기 변경
position = driver.get_window_position()
x = position.get('x')
y = position.get('y')

print("x : "+str(x)+" "+"y : "+str(y))

driver.maximize_window()

driver.fullscreen_window()
'''


###### 스크롤 내리기 #########
print(keywords + '스크롤 중 ......')
elem = driver.find_element_by_tag_name('body')
for i in range(60):
    elem.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.2)

try:
    # ////*[@id="islmp"]/div/div/div/div[2]/div[1]/div[2]/div[2]/input
    driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div[2]/div[1]/div[2]/div[2]/input').click()
    for i in range(60):
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)
except:
    pass


### 이미지 개수 파악 ########

links = []
images = driver.find_elements_by_css_selector("img.rg_i.Q4LuWd")

for image in images:
    if image.get_attribute('src') != None:
        links.append(image.get_attribute('src'))

    # base 64 는 이미지파일을 전송하거나 불러올때 이진으로 인코딩한거 . 컴퓨터가 읽기 편하게 만든게 base64이다.
print(keywords + '찾은 이미지 개수 :', len(links))
time.sleep(2)

#### 데이터 다운로드 ####
folder_name ='./' + keywords + '_img_download'
create_folder(folder_name)
for index, i in enumerate(links):
    url = i
    start = time.time()
    urllib.request.urlretrieve(
        url, folder_name + '/' + keywords + '_' + str(index) + '.jpg')
    print(str(index) + '/' + str(len(links)) + ' ' + keywords + ' 다운로드 시간 ------ :', str(time.time() - start)[:5] + '초')
    # print(keywords + '다운로드 완료 !!')