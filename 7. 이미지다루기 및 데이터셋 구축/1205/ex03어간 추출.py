'''
# 어간 추출 : 단어의 어간을 구분하여 기본 의미를 유지하면서 어미를 제거합니다. 
- 텍스트 데이터에서 어간을 추출하면 읽기는 힘들어지지만 기본의미에 가까워지고 샘플 간에 비교하기 더 좋음
- NLTK 의 PorterStemmer 는 단어의 어미를 제거하여 어간을 바꿀 수 있습니다.
'''
from nltk.stem.porter import PorterStemmer

# 단어 토큰을 만듭니다.
tokenized_words = ['i', 'am', 'humbled', 'by', 'this', 'traditional', 'meeting']

# 어간 추출기를 만듭니다.
porter = PorterStemmer()

word_list = []
# 어간 추출기를 적용합니다.
for word in tokenized_words:
    word_list.append(porter.stem(word))

print(word_list) # 추출된 어간 프린트