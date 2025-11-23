import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from adore import ADORE
from adore.storytelling import generate_abt_story, select_top_k_feature_contributions
from adore.visualizing import plot_abt_story, plot_feature_contributions


# Utility function for scaling features
def scale_features(scaler, data, feature_names):
    scaled_data = scaler.transform(data)
    return pd.DataFrame(scaled_data, columns=feature_names)


# Step 1: Load the dataset and extract features and target variable
try:
    data = pd.read_csv('../data/advertising.csv')  # Replace with your data file path
    assert all(col in data.columns for col in ['TV', 'radio', 'newspaper', 'sales']), \
        "Dataset must contain columns: 'TV', 'radio', 'newspaper', 'sales'"
except Exception as e:
    raise ValueError(f"Error loading dataset: {e}")

X = data[['TV', 'radio', 'newspaper']]  # Features
y = data['sales']  # Target variable

# Step 2: Standardize the features
scaler = StandardScaler()
scaler.fit(X)
X_scaled_df = scale_features(scaler, X, X.columns)

# Step 3: Train the original linear regression model
model_simple = LinearRegression()
model_simple.fit(X_scaled_df, y)


# Step 4: Define the custom non-linear model (ComplexModel)
class ComplexModel:
    def fit(self, X, y):
        pass  # This model does not require fitting

    def predict(self, X):
        TV = X[:, 0]
        radio = X[:, 1]
        newspaper = X[:, 2]
        return 3 * TV ** 2 + 2 * newspaper * radio + radio + 2


model_complex = ComplexModel()

# Step 5: Select new input features for explanation
Samples_X = np.array([[1260.1, 87.8, 69.2],
                      [354.5, 55.3, 45.1],
                      [23.5, 34.3, 27.1],
                      [154.5, 35.3, 35.1]])

# Step 6: Standardize the new input features and keep feature names
Samples_X_scaled_df = scale_features(scaler, Samples_X, X.columns)

# 检查标准化后数据范围是否合理
print("Samples_X_scaled_df:\n", Samples_X_scaled_df.describe())

# Step 8: Use ADORE to explain the predictions of the complex model
adore = ADORE(model=model_complex, data=Samples_X_scaled_df.values)
results = adore.explain()

contributions = results[0]  # Feature contribution matrix (samples x features)
feature_names = X.columns.tolist()
print("Contributions Matrix Shape:", contributions.shape)

# 生成 top-k 数据
top_k_df = select_top_k_feature_contributions(contributions, feature_names, k=5)
print("Top-k Contributions DataFrame:\n", top_k_df)

# 生成 ABT 文本
sample_indices = range(Samples_X_scaled_df.shape[0])
abt_story = generate_abt_story(
    feature_contributions=contributions,
    X_index=sample_indices,
    feature_names=feature_names,
    top_k_contributions_df=top_k_df,
    model_name='ComplexModel',
    language="english"
)
and_part, but_part, therefore_part = abt_story

# 绘制特征贡献（单独展示）
plot_feature_contributions(contributions, feature_names=feature_names, model_name="ComplexModel")

# 可视化 ABT 文本
plot_abt_story(abt_story, model_name="ComplexModel")
