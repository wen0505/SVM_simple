"""
Example 1 (簡單線性回歸)
先從簡單的線性回歸舉例， y=ax+b ， a 稱為斜率， b 稱為截距。
"""
# imports
import numpy as np
import matplotlib.pyplot as plt

# 亂數產生資料
np.random.seed(0)
noise = np.random.rand(100, 1)
x = np.random.rand(100, 1)
y = 3 * x + 15 + noise
# y=ax+b Target function  a=3, b=15

# plot
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
