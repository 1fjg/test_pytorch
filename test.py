import matplotlib.pyplot as plt
import numpy as np
import torch
from d2l import torch as d2l
# 定义数据
x = np.linspace(-5, 5, 1400)
params = [(0, 1), (1, 2), (2, 3)]

def normal(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# 使用 d2l.plot 直接绘制
y = np.array([normal(x, mu, sigma) for mu, sigma in params])

# 绘图
d2l.plot(x, y, xlabel='x', ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
plt.show()