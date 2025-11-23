import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from adore import ADORE
from adore.storytelling import generate_freytag_story, select_top_k_feature_contributions
from adore.visualizing import plot_freytag_stories

# Step 1: Load the dataset and extract features and target variable
data = pd.read_csv('../data/advertising.csv')  # Replace with your data file path
X = data[['TV', 'radio', 'newspaper']]  # Features
y = data['sales']  # Target variable

# Step 2: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)  # Keep the feature names
print("X_scaled:", X_scaled_df.head())

# Step 3: Train the original linear regression model
model_simple = LinearRegression()
model_simple.fit(X_scaled, y)

# Step 4: Define the custom non-linear model (ComplexModel)
class ComplexModel:
    """
    This is a custom model implementing the formula:
    y = 3 * TV^2 + 2 * Newspaper * Radio + Radio + 2
    """
    def predict(self, X):
        TV = X[:, 0]
        radio = X[:, 1]
        newspaper = X[:, 2]
        return 3 * TV**2 + 2 * newspaper * radio + radio + 2

model_complex = ComplexModel()

# Step 5: Select new input features for explanation
Samples_X = np.array([[1260.1, 87.8, 69.2],
                      [354.5, 55.3, 45.1],
                      [23.5, 34.3, 27.1],
                      [154.5, 35.3, 35.1]])

# Step 6: Standardize the new input features and keep feature names
Samples_X_scaled = scaler.transform(Samples_X)
Samples_X_scaled_df = pd.DataFrame(Samples_X_scaled, columns=X.columns)  # Keep the feature names
print("Samples_X_scaled:", Samples_X_scaled_df.head())

# Step 7: Make predictions with the simple model
predictions_simple = model_simple.predict(Samples_X_scaled)
print("Predictions for Samples_X (Simple Model):", predictions_simple)

# Step 8: Use ADORE to explain the predictions of the complex model
adore = ADORE(model=model_complex, data=Samples_X_scaled)
results = adore.explain()
contributions = results[0]  # Get the feature contributions matrix
print("----the contributions matrix starts-------")
print(contributions)
print("----the contributions matrix ends-------")

# Step 9: Get the feature names
feature_names = X.columns.tolist()  # Get the list of feature names

# Step 10: Automatically select Top-k contributions
top_k_df = select_top_k_feature_contributions(contributions, feature_names, k=5)

# Step 11: Generate Freytag story based on contributions
freytag_story = generate_freytag_story(contributions, range(len(Samples_X_scaled)), feature_names, top_k_df, model_name='ComplexModel')

# Display the generated stories
print(freytag_story)

# Step 12: Visualize Freytag pyramid structure
plot_freytag_stories(freytag_story, top_k_df)
