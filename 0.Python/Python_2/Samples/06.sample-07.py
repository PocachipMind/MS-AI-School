import requests
from bs4 import BeautifulSoup
# 웹크롤링 해주는 뷰티풀수프

def get_exchange_rate(target1, target2):
  headers = {
      'User-Agent' : 'Mozilla/5.0',
      'Content-Type' : 'text/html; charest=utf-8'
  }

  response = requests.get('https://kr.inversting.com/currencies/{}-{}'.format(target1,target2),headers = headers)

  content = BeautifulSoup(response.content, 'html.parser')
  containers = content.find('span' , {'data-test' : 'instrument-price-last'})

  print(containers.text)

get_exchange_rate('usd','krw')
