# 딕셔너리 특성 행렬로 변환
# DictVectorizer 클래스는 0이 아닌 것의 원소만 저장하는 희소 행렬을 반환
# 자연어 처리 분야와 같은 매우 큰 행렬을 다룰 때 메모리 사용량을 최소화해야하기 때문에 유용합니다.
# parse=False로 지정하면 밀집 벡터를 출력할 수 있습니다.
# get_feature_names()를 사용하여 생성된 특성의 이름을 얻을 수 있습니다.
from sklearn.feature_extraction import DictVectorizer
import pandas as pd

data_dict = [{"Red": 2, "Blue": 4},
             {"Red": 4, "Blue": 3},
             {"Red": 1, "Yellow": 2},
             {"Red": 2, "Yellow": 2}
             ]
# print(data_dict)
# [{'Red': 2, 'Blue': 4}, {'Red': 4, 'Blue': 3}, {'Red': 1, 'Yellow': 2}, {'Red': 2, 'Yellow': 2}]

# 0이 아닌 원소만 저장하는 희소 행렬을 반환
dictvectorizer = DictVectorizer(sparse=False)

# 딕셔너리를 특성 행렬로 변환
features = dictvectorizer.fit_transform(data_dict)
features_names = dictvectorizer.get_feature_names() # 특성 이름 획득 가능
print(features_names)
dict_data = pd.DataFrame(features, columns=features_names)
print(dict_data)

## 행렬 형태로 만들기
# 네 개의 문서에 대한 단어 카운트 딕셔너리를 만듭니다.
# 단어 카운트 딕셔너리 만들기
doc_1_word_counter = {"Red": 2, "Blue": 4}
doc_2_word_counter = {"Red": 4, "Blue": 4}
doc_3_word_counter = {"Red": 1, "Yellow": 2}
doc_4_word_counter = {"Red": 2, "Yellow": 2}
doc_word_counts = [doc_1_word_counter, doc_2_word_counter, 
                   doc_3_word_counter, doc_4_word_counter]

print(doc_word_counts)
# 단어 카운트 딕셔너리를 특성 행렬로 변환합니다.
data_array = dictvectorizer.fit_transform(doc_word_counts)
print(data_array)