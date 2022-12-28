import numpy as np
import matplotlib.pyplot as plt

# [-10 ~ 10] 구간에서 100개의 t값을 시그모이드 함수에 대입합니다.
t = np.linspace(-10, 10, 100)
print(t)

# 시그모이드 공식
# np.exp -> 함수 밑(base) 이 자연상수 e 인 지수함수로 변환
sig = 1 / (1 + np.exp(-t))

# t 와 시그모이드 결과 그래프
plt.figure(figsize=(9, 3))
plt.plot([-10, 10], [0, 0], "k-")
plt.plot([-10, 10], [0.5, 0.5], "k:")  # y축 기준 0.5 실선 하나 생성
plt.plot([-10, 10], [1, 1], "k:")  # y축 기준 1 실선 하나 생성
plt.plot([0, 0], [-1.1, 1.1], "k-")  # center 기준 0.0 선하나 생성
plt.plot(t, sig, "r-", linewidth=2,
         label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
plt.xlabel("t")
plt.legend(loc="upper left", fontsize=20)
plt.axis([-10, 10, -0.1, 1.1])  # 그래프 간격
plt.show()
