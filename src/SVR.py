"""
Example 1 (簡單線性回歸)
scikit-learn KNN迴歸模型的score函式是R2 score，可作為模型評估依據，其數值越接近於1代表模型越佳。
除了R2 score還有其他許多回歸模型的評估方法，例如： MSE、MAE、RMSE。
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics

# 亂數產生資料
np.random.seed(0)
noise = np.random.rand(100, 1)
x = np.random.rand(100, 1)
y = 3 * x + 15 + noise
# y=ax+b Target function  a=3, b=15

# 建立SVR模型
linearModel = svm.SVR(C=1, kernel='linear')
# 使用訓練資料訓練模型
linearModel.fit(x, y)
# 使用訓練資料預測
predicted = linearModel.predict(x)

print('R2 score: ', linearModel.score(x, y))
mse = metrics.mean_squared_error(y, predicted)
print('MSE score: ', mse)

# plot
plt.scatter(x, y, s=10, label='True')
plt.scatter(x, predicted, color="r", s=10, label='Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


