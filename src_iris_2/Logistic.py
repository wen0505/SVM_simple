import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# (1) 載入資料集
iris = load_iris()
df_data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                       columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])
df_data

# (2) 切割訓練集與測試集
X = df_data.drop(labels=['Species'], axis=1).values     # 移除Species並取得剩下欄位資料
y = df_data['Species'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print('train shape:', X_train.shape)
print('test shape:', X_test.shape)

# 建立Logistic模型
logisticModel = LogisticRegression(multi_class='auto', solver='newton-cg', random_state=0)
# 使用訓練資料訓練模型
logisticModel.fit(X_train, y_train)
# 使用訓練資料預測分類
predicted = logisticModel.predict(X_train)

# (3) 使用Score評估模型
# 預測成功的比例
print('訓練集: ', logisticModel.score(X_train, y_train))
print('測試集: ', logisticModel.score(X_test, y_test))

# (4) 真實分類
# 建立測試集的 DataFrame
df_test = pd.DataFrame(X_test, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
df_test['Species'] = y_test
pred = logisticModel.predict(X_test)
df_test['Predict'] = pred

sns.lmplot(x="PetalLengthCm", y="PetalWidthCm", hue='Species', data=df_test, fit_reg=False, legend=False)
plt.legend(title='target', loc='upper left', labels=['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica'])
plt.title('real result')     # wen
plt.show()

# (5) Logistic regression (訓練集)預測結果
sns.lmplot(x="PetalLengthCm", y="PetalWidthCm", data=df_test, hue="Predict", fit_reg=False, legend=False)
plt.legend(title='target', loc='upper left', labels=['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica'])
plt.title('Logistic regression result')     # wen
plt.show()
