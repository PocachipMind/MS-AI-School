'''
# 단어 중요도에 가중치 부여하기
- TfidVectorizer : tf-idf(단어 비도 - 역문서 빈도)를 사용해 트윗, 영화 리뷰, 연설문 등 하나의 문서에 등장하는
단어의 빈도와 다른 모든 문서에 등장하는 빈도를 비교합니다.
- tf(단어 빈도) : 한 문서에 어떤 단어가 많이 등장할수록 그 문서에 더 중요한 단어
- df(문서 빈도) : 한 단어가 많은 문서에 나타나면 이는 어떤 특정 문서에 중요하지 않는 단어
- tf 와 df 두 통계치를 연결하여 각 문서가 문서에 얼마나 중요한 단어인지를 점수로 할당
- tf 를 idf (역문서 빈도)에 곱합니다.
'''

# 단어 중요도에 가중치 부여하기
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])

# tf-idf 특성 행렬을 만듭니다.
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)
feature_matrix.toarray() # tf-idf 특성 행렬을 밀집 배열로 확인
print(feature_matrix) # tf-idf 특성 행렬 확인

# 특성 이름을 확인
tf = tfidf.vocabulary_
print("...",tf)

'''
결과
{'love': 6, 'brazil': 3, 'sweden': 7, 'is': 5, 'best': 1, 'germany': 4, 'beats': 0, 'both': 2}
'''
