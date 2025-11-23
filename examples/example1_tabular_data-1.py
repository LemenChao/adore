# -*- coding: utf-8 -*-
"""
Module Name: example1_tabular_data-1.py
Description:
    This module takes the advertisement dataset as an example, predicts advertisement features using linear regression
    models and custom nonlinear complex models, and interprets and analyzes the prediction results using the ADORE
    algorithm. Meaningful results were also demonstrated by visualizing the functional functions.

    Key functionalities include:
        - **Model Training and Predictions:** Training a CNN model on scaled data and generating predictions using various machine learning models (e.g., Linear Regression, Random Forest, SVR, and CNN).
        - **Model Explanation with ADORE:** Leveraging the ADORE framework to explain predictions from multiple models, providing detailed feature contributions for samples.
        - **Visualization Tools:** A suite of visualization functions for:
        - **Feature Contribution Analysis:** Plotting feature importance for different models.
        - **Sample Contribution Analysis:** Understanding contributions at the individual sample level.
        - **Heatmaps:** Visualizing feature contributions across samples in a comparative format.
        - **Algorithm Comparison:** Comparing feature contributions between models using radar charts and other comparative visuals.

Author:
    Fang Anran <fanganran97@126.com>

Maintainer:
    Fang Anran <fanganran97@126.com>

Contributors:
    Chao Lemen <chaolemen@ruc.edu.cn>
    Lei Ming <leimingnick@ruc.edu.cn>
    Fang Anran <fanganran97@126.com>

Created on: 2024-12-25
Last Modified on: 2024-12-31
Version: [0.0.1]

License:
    This module is licensed under GPL-3.0 . See LICENSE file for details.
"""
# --------------------------------

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from adore import ADORE
from adore.visualizing import *

# Set up logger
logger = logging.getLogger(__name__)

# 1. Step 1: Read the data
data = pd.read_csv('../data/advertising.csv')  # Replace with the path to your data file
X = data[['TV', 'radio', 'newspaper']]  # 3Features
y = data['sales']  # Target variable

# 2. Step 2: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)  # Keep the feature names
print("X_scaled:", X_scaled_df.head())

# 3. Step 3: Train the original linear regression model
model_simple = LinearRegression()
model_simple.fit(X_scaled, y)

# 4. Define the simple non-linear model
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

# 5. Step 4: Select new input features for explanation
Samples_X = pd.DataFrame([
    [1260.1, 87.8, 69.2],
    [354.5, 55.3, 45.1],
    [23.5, 34.3, 27.1],
    [154.5, 35.3, 35.1]
], columns=X.columns)  # Keep the feature names directly in DataFrame


# 6. Step 5: Standardize the new input features and keep feature names
Samples_X_scaled = scaler.transform(Samples_X)
Samples_X_scaled_df = pd.DataFrame(Samples_X_scaled, columns=X.columns)  # Keep the feature names
logger.info(f"Samples_X_scaled: {Samples_X_scaled_df.head()}")

# 7. Step 6: Make predictions with the simple model
predictions_simple = model_simple.predict(Samples_X_scaled)
logger.info(f"Predictions for Samples_X (Simple Model):{predictions_simple}")

# 8. Step 7: Use ADORE to explain the predictions
# adore = ADORE(model=model_complex, X=Samples_X_scaled)
adore = ADORE(model=model_complex, data=Samples_X)
results = adore.explain()

# 9. Step 8: Retrieve and display the explanation results
contributions = results[0]  # Feature contribution matrix
derivative_matrix = results[1]
EACM = results[2]
ERCM = results[3]
SACM = results[4]
SRCM = results[5]
FACM = results[6]
FRCM = results[7]

# Output explanation results
logger.info("Feature contributions for each sample (Simple Model):")
print(contributions)
logger.info("Absolute and relative contribution of the samples")
print(SACM, SRCM, sep='\n')
logger.info("Absolute and relative contribution of the features")
print(FACM, FRCM, sep='\n')

# # 10. Step 9: Visualize the explanation results
plot_feature_contributions(contributions, X.columns.tolist())
plot_sample_contributions(contributions, X.columns.tolist())
plot_contributions_heatmap(contributions, X.columns.tolist())
plot_violin_chart(contributions, X.columns.tolist())
plot_radar_chart(contributions, X.columns.tolist())
plot_bubble_chart(contributions, Samples_X_scaled_df, X.columns.tolist(), size_factor=2)
plot_explained_variance_curve([0.2, 0.5, 0.7, 0.9, 1.0])
plot_custom_contribution_summary(contributions, Samples_X_scaled_df, X.columns.tolist())
