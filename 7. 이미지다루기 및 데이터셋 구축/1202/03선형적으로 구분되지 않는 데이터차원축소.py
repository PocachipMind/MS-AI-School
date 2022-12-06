# 선형적으로 구분되지 않는 데이터차원 축소
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles

# 선형적으로 구분되지 않는 데이터를 만듭니다.
features, _ = make_circles(
    n_samples=1000, random_state=1, noise=0.1, factor=0.1)
print(features)

# 방사 기저 함수 (radius basis function, RBF)를 사용하여 커널 PCA 적용
kpca = KernelPCA(kernel="rbf", gamma=15, n_components=1)
features_kpca = kpca.fit_transform(features)

print("원본 특성 개수 : ", features.shape[1])
print("줄어든 특성 개수 : ", features_kpca.shape[1])


# 커널 함수는 선형적으로 구분되지 않는 데이터를 선형적으로 구분되는 고차원으로 투영시켜줍니다. (커널 트릭)
# 사이킷런의 KernelPCA의 kernel 매개변수의 값 - rbf(가우시안 방사 기저 함수 커널), ploy(다항식 커널), sigmoid(시그모이드 커널), 선형 투영(linear)로 지정
# 여러 가지 커널과 매개변수 조합으로 머신러닝 모델을 여러 번 훈련시켜서 가장 높은 예측 성능을 만드는 값의 조합을 찾아야 합니다.
# 커널 트릭은 실제 고차원으로 데이터를 변환하지 않으면서 고차원 데이터를 다루는 듯한 효과를 냅니다.
