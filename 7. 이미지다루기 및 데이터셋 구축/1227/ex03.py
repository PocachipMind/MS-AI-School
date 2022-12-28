# b값이 변화에 따른 좌 우 이동
# b값에 따라서 그래프가 어떻게 변하는지확인 !!
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(t):
    return 1/(1 + np.exp(-t))


x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x+0.5)
y2 = sigmoid(x+1)
y3 = sigmoid(x+1.5)

plt.plot(x, y1, 'r', linestyle='--')  # x + 0.5인 경우
plt.plot(x, y2, 'g')  # x + 1 인경우
plt.plot(x, y3, 'b', linestyle='--')  # x + 1.5 인경우
plt.plot([0, 0], [1.0, 0.0], ':')  # 가운데 점선 추가
plt.title('sigmoid Function')
plt.show()

# 양상(존재여부) 범주
# 0(음성) if p < 0.5
# 1(양성) if p >= 0.5
# 결론적으로는 t>= 0 이면 sigma(t) >= 0.5 이므로 (반대는 반대로)
# 시그모이드 공식과 함께 생각하면 θ^T·x가 양수일때 1(양성 범주), 음수일때 0(음성 범주) 이라고 예측한다.
