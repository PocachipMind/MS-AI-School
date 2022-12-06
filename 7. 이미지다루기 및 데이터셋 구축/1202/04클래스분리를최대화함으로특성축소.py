'''
• 선형 판별 분석(LDA)를 사용하여 클래스를 최대한 분리하는 성분 축으로 특성을 투영합니다.
• LDA는 분류 알고리즘이지만 차원 축소에도 자주 사용되는 기법
• LDA는 특성 공간을 저차원 공간으로 투영합니다
• explained_variance_ratio_ 속성에서 각 성분이 설명하는 분산의 양을 확인할 수 있습니다.
'''

from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()  # 붓꽃 데이터셋을 로드
features = iris.data
target = iris.target

# LAD 객체 만들고 실행하여 특성을 변환합니다 .
lda = LinearDiscriminantAnalysis(n_components=None)
features_lad = lda.fit(features, target)

# print("원본 특성 개수 : ", features.shape[1])
# print("줄어든 특성 개수 : ", features_lad.shape[1])

# 설명된 분산의 비율이 담긴 배열을 저장
lda_var_ratios = lda.explained_variance_ratio_
print(lda_var_ratios)


def select_n_components(var_ratio, goal_var: float) -> int:
    total_variances = 0.0  # 설명된 분산의 초기값을 지정
    n_components = 0  # 특성 개수의 초기값을 지정

    for explained_variance in var_ratio:  # 각 특성 의 성명된 분산을 순회 Loop
        total_variances += explained_variance  # 설명된 분산 값을 누적
        n_components += 1  # 성분 개수를 카운트
        if total_variances >= goal_var:  # 설명된 분산이 목표치에 도달하면 반복을 종료
            break
    return n_components  # 성분 개수를 반환


temp = select_n_components(lda_var_ratios, 0.95)
print("temp = ", temp)
