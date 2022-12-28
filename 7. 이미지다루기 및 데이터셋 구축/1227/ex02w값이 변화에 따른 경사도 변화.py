# w값이 변화에 따른 경사도 변화
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(t):
    return 1/(1 + np.exp(-t))


x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5 * x)
y2 = sigmoid(x)
y3 = sigmoid(2*x)

print("y1: ", y1)
print("y2: ", y2)
print("y3: ", y3)

plt.plot(x, y1, 'r', linestyle='--')  # w 0.5인 경우
plt.plot(x, y2, 'g')  # w 1 인경우
plt.plot(x, y3, 'b', linestyle='--')  # w 2인경우
plt.plot([0, 0], [1.0, 0.0], ':')  # 가운데 점선 추가
plt.title('sigmoid Function')
plt.show()
