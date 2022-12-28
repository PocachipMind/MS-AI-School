import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드
from sklearn import datasets

# 로지스틱 회귀 모델 훈련
from sklearn.linear_model import LogisticRegression

# iris data
iris = datasets.load_iris()
list_iris = []

# dict keys 무엇인지 체크 필요
list_itis = iris.keys()
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x = iris["data"][:, 3:]  # 꽃잎의 너비 변수 사용
print(iris["target_names"])  # ['setosa' 'versicolor' 'virginica']
y = (iris["target"] == 2).astype("int")  # ris-versinica 면 1 아니면 0
print(y)

log_reg = LogisticRegression(solver="liblinear")
# log_reg = LogisticRegression(solver='liblinear' , random_state = 42)
# 사이킷런의 LogisticRegression은 클래스 레이블을 반환 하는 predict() 메서드
# 클래스에 속할 확률을 반환하는 predict_proba() 메서드를 가지고 있습니다.
# predict() 메서드는 확률 추정식에서 0보다 클 때를 양성클래스로 판단하여 결과를 반환하고
# predict_proba 메서드는 시그모이드 함수를 적용하여 계산한 확률을 반환합니다.
log_reg.fit(x, y)

# 이제 꽃잎이 너비가 0 ~ 3CM 곷에 대해 모델의 추정 확률을 계산
# x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
# # -1의 의미는 변경된 배열의 -1위치의 차원은 원래 배열의 길이와 남은 차원으로 부터 추청된다라는 뜻입니다.
# y_proba = log_reg.predict_proba(x_new)
# plt.plot(x_new, y_proba[:, 1], 'g-', label='Iris-Virginica')  # 음성
# plt.plot(x_new, y_proba[:, 0], 'b--', label='Not Iris-Virginica')  # 양성
# plt.legend
# plt.show()

# 좀더 보기 좋게 표기하기
x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(x_new)
decision_boundary = x_new[y_proba[:, 1] >= 0.5][0]

plt.figure(figsize=(8, 3))  # 그래프 사이즈
plt.plot(x[y == 0], y[y == 0], "bs")  # 음성 범주 pointing
plt.plot(x[y == 1], y[y == 1], "g^")  # 양성 범주 pointing

# 결정경계 표시
plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)

# 추정확률 plotting
plt.plot(x_new, y_proba[:, 1], 'g-', label='Iris-Virginica')  # 음성
plt.plot(x_new, y_proba[:, 0], 'b--', label='Not Iris-Virginica')  # 양성

# 결정 경계 표시
plt.text(decision_boundary+0.02, 0.15, "Decision boundary",
         fontsize=14, color="k", ha="center")
plt.arrow(decision_boundary, 0.08, -0.3, 0,
          head_width=0.05, head_length=0.1, fc='b', ec='b')
plt.arrow(decision_boundary, 0.92, 0.3, 0,
          head_width=0.05, head_length=0.1, fc='g', ec='g')
plt.xlabel("petal width(cm)", fontsize=14)
plt.ylabel("probability", fontsize=14)
plt.legend(loc='center left', fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
plt.show()





# 보기좋게 다시 재정의
x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(x_new)
decision_boundary = x_new[y_proba[:, 1] >= 0.5][0]

# 그러면 결정경계가 어떤 값을 가지고 분류하는가?
# 양쪽의 확률이 50%가 되는 1.6cm 근방에서 결정경계가 만들어지고 분류기는 1.6cm 보다 크면 Iris virginica 분류하고 
# 작으면 아니라고 예측

# 그렇게 되는가 ?
test_data = log_reg.predict([[1.8], [1.48]])

# 예상 1.6 기준 - 1.8 true, 1.48 false >> [1, 0] 나올것임
print(f"진짜 우리가 원하는 분류가 되는가? 분류기준 {decision_boundary} result -> {test_data}") 
# 진짜 우리가 원하는 분류가 되는가? 분류기준 [1.61561562] result -> [1 0]
