from adore import ADORE
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 加载 Iris 数据集
X, y = load_iris(return_X_y=True, as_frame=True)
feature_names = X.columns

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练随机森林模型
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# 模型评估
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"模型在测试集上的准确率:{acc:.4f}")

from adore.visualizing import plot_feature_contributions

# ------- 原始特征版本 -------
explainer_raw = ADORE(model=model, data=X)
contributions_raw, _, _, _, _, _, _, _ = explainer_raw.explain()
plot_feature_contributions(contributions_raw, feature_names=feature_names, model_name="randomforest-raw")

# ------- 标准化版本（更稳健的扰动尺度） -------
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
# explainer_std = ADORE(model=model, data=X_scaled_df[1:5], delta_method='mad', epsilon=0.1)
# contributions_std, _, _, _, _, _, _, _ = explainer_std.explain()
# plot_feature_contributions(contributions_std, feature_names=feature_names, model_name="randomforest-standardized")
