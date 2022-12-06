'''
# 텍스트 토근화
- 텍스트를 개별 단어로 나눕니다.
- 자연어 처리 툴킷인 NLTK는 단어 토큰화를 비롯해 강력한 텍스트 처리 기능을 제공
'''
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import nltk

# 구두점 데이터를 다운로드 합니다.
nltk.download('punkt')

# 텍스트 데이터 생성
string_temp = "The science of today is the technology of tomorrow"
# 단어를 토큰으로 나눕니다.
token_temp = word_tokenize(string_temp)
print(token_temp)

string_temp01 = "The science of today is the technology of tomorrow. Tomorrow id today. Is day."
sent_data = sent_tokenize(string_temp01)
print(sent_data)