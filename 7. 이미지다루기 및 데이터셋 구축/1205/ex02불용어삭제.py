'''
# 불용어 삭제
- 불용어는 작업 전에 삭제해아하는 일련의 단위를 의미하기도 하지만 유용한 정보가 거의 없는 매우 자주 등장하는
단어를 의미합니다.
- NLTK 의 stopwords (불용어 리스트 179개)를 사용하여 토큰화된 단어에서 불용어를 찾고 삭제할 수 있습니다.
- NLTK 의 stopwords 토큰화된 단어가 소문자라고 가정합니다.
- 사이킷런도 영어 불용어 리스트를 제공합니다 (318개)
- 사이킷런도 불용어는 frozenset 객체이기 때문에 인덱스를 사용할 수 없습니다
'''
# 불용어 삭제
from nltk.corpus import stopwords
import nltk

# 불용어 데이터를 다운로드 -> 179개 입니다.
nltk.download('stopwords')

# 단어 토큰을 만듭니다.
tokenized_words = ['i', 'am', 'the', 'of', 'to', 'go', 'store', 'and', 'park']

# 불용어 로드
stop_words = stopwords.words('english')
print("불용어 리스트 길이 >>",len(stop_words)) # 179개
print("불용어 리스트 >>", stop_words)

# 불용어 삭제
for word in tokenized_words:
    if word not in stop_words:
        print(word)
#[word for word in tokenized_words if word not in stop_words]

#stop_data = stop_words # 불용어 확인
#print("불용어 확인", stop_data)
