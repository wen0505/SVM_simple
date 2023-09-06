"""
Example 2 (非線性回歸)
現實的資料可能並只會發生在一條線上。下面示範一個多項次的回歸模型例子。

kernel='rbf'

Parameters:
    C: 限制模型的複雜度，防止過度擬合。
    kernel: 此範例採用 Radial Basis Function 高斯轉換。
    gamma: 數值越大越能做複雜的分類邊界。
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics

x = np.array([[1.40280301e-01], [9.03857692e-01], [5.35815131e-01], [3.58391981e-01], [2.43418162e-02],
              [2.43342904e-02], [3.37370600e-03], [7.50261116e-01], [3.61339257e-01], [5.01366775e-01],
              [4.23721405e-04], [9.40725121e-01], [6.92960750e-01], [4.50878979e-02], [3.30603187e-02],
              [3.36372142e-02], [9.25633424e-02], [2.75369313e-01], [1.86576499e-01], [8.48144121e-02],
              [3.74363965e-01], [1.94585372e-02], [8.53484957e-02], [1.34221000e-01], [2.07999831e-01],
              [6.16501290e-01], [3.98696193e-02], [2.64437058e-01], [3.50955021e-01], [2.15764084e-03],
              [3.69110747e-01], [2.90784768e-02], [4.23170975e-03], [9.00383763e-01], [9.32445223e-01],
              [6.53506272e-01], [9.27895484e-02], [9.53984185e-03], [4.68174835e-01], [1.93734218e-01]])

y = np.array([ 5.82469676e+00,  7.94613194e+00,  9.24976070e+00,  6.59761731e+00,  2.16651685e+00,
               -2.50365745e-03, -1.00182588e+00, 9.02075194e+00,  8.57086436e+00,  8.50848958e+00,
               -7.34549241e-02, 8.73802779e+00,  7.26038154e+00,  2.38778217e+00,  2.02397265e+00,
               3.57417666e+00,  5.15052189e+00,  5.57291682e+00,  6.83461431e+00,  4.20408429e+00,
               7.21499207e+00,  2.24057093e+00,  5.63575746e+00,  6.66180813e+00,  5.91402744e+00,
               8.29511673e+00,  3.18174801e+00,  8.23158707e+00,  7.30330971e+00,  2.55480191e-02,
               6.76197223e+00,  1.05656839e+00,  1.21851645e+00,  1.03566236e+01,  8.95941549e+00,
               9.67640393e+00,  5.17463285e+00,  2.25781800e-01,  8.60137397e+00,  8.13359834e+00])

# 建立 kernel='rbf' 模型
rbfModel = svm.SVR(C=6, kernel='rbf', gamma='auto')
# 使用訓練資料訓練模型
rbfModel.fit(x, y)
# 使用訓練資料預測分類
predicted = rbfModel.predict(x)
# 計算訓練集 MSE 誤差
mse = metrics.mean_squared_error(y, predicted)
print('訓練集 MSE: ', mse)
x_test = np.linspace(-0.1, 1.1, 500)[:, None]
predicted = rbfModel.predict(x_test)
plt.scatter(x.ravel(), y)
plt.plot(x_test, predicted, label='kernel=linear', color='r')
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
