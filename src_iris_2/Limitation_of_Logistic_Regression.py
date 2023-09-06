"""
Logistic 其實有很大的限制，如下圖範例在線性不可分的時候無法有效預測。我們可以發現無法一刀可以將這四筆資料分成兩個類別。
因為在剛剛的計算方法 Logistic Regression 在兩個類別中僅會切出一條直線。以上例來看不管怎麼分割，始終無法將資料分離出來。
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from plot_decision_regions import PLOT_DECISION_REGIONS
from plot_hyperplane import PLOT_HYPERPLINE

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = [1, 0, 0, 1]

# 建立Logistic模型
logisticModel = LogisticRegression(random_state=0)
# 使用訓練資料訓練模型
logisticModel.fit(X, y)
# 使用訓練資料預測分類
logisticModel.predict(X)

color = "rb"
color = [color[y[i]] for i in range(len(y))]
plt.scatter(X[:, 0], X[:, 1], c=color)
plt.show()

# 繪製決策邊界
plt.figure()
PLOT_DECISION_REGIONS(X, y, logisticModel)
# Plot also the training points
colors = "rb"
for i, color in zip(logisticModel.classes_, colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired)

PLOT_HYPERPLINE(0, 'r')
plt.show()
