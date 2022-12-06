'''
# 품사 태킹
- NLTK는 품사 태그를 위해 구문 주석 말뭉치인 펜 트리뱅크(Eenn Treebank)를 사용합니다.
- 텍스트가 태깅되면 태그를 사용해 특정 품사를 찾을 수 있습니다
'''
import nltk
from nltk import pos_tag
from nltk import word_tokenize

# 태거를 다운로드
nltk.download('averaged_perceptron_tagger')
# 샘플 텍스트 데이터
text_data = 'Chris loved outdoor running'

# 사전 훈련된 품사 태깅을 사용합니다.
text_tagged = pos_tag(word_tokenize(text_data))
print(text_tagged)
'''
결과
[('Chris', 'NNP'), ('loved', 'VBD'), ('outdoor', 'RP'), ('running', 'VBG')]
'''