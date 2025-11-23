# -*- coding: utf-8 -*-
"""
Module Name: example2_tabular_data-2.py
Description:
    This module simulates the entire process of training, predicting, and interpreting a machine learning model on
    tabular data, using the Boston house price dataset as an example, demonstrating the effectiveness of the ADORE algorithm for in-depth analysis and evaluation of a model's performance. Finally, the feature and sample-level contributions are revealed, and visualizations help to more intuitively understand the decision logic and important drivers of the model.

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
    Chaolemen Borjigin <chaolemen@ruc.edu.cn>
    Lei Ming <leimingnick@ruc.edu.cn>
    Fang Anran <fanganran97@126.com>

Created on: 2024-12-25
Last Modified on: 2024-12-25
Version: [0.8.0]

License:
    This module is licensed under GPL-3.0 . See LICENSE file for details.
"""
# --------------------------------

from keras.layers import Dense, Conv1D, Flatten
from keras.models import Sequential
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from adore import ADORE
from adore.visualizing import *

# Set up logger
logger = logging.getLogger(__name__)

# 1. Step 1: Read the data
boston = pd.read_csv('../data/housing.csv')  # Replace with the path to your data file

X = boston[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
y = boston['MEDV']

# 2. Step 2: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)  # Keep the feature names
print("X_scaled:", X_scaled_df.head())

# 3. Step 3: Train three models
# Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_scaled, y)

# Random Forest Regressor
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_scaled, y)

# Support Vector Regressor
model_svr = SVR(kernel='linear')
model_svr.fit(X_scaled, y)

# Convolutional Neural Network (CNN) for regression
model_cnn = Sequential([
    Conv1D(filters=16, kernel_size=2, activation='relu', input_shape=(X_scaled.shape[1], 1)),
    Flatten(),
    Dense(10, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the CNN model
model_cnn.compile(optimizer='adam', loss='mean_squared_error')

# Fit the CNN model
model_cnn.fit(X_scaled, y, epochs=100, batch_size=32, verbose=1)

# 4. Step 4: Select sample for explanation
indices = [35, 358, 40, 203, 391, 289, 155, 370, 363, 142]  # Randomly select 10 indexes that do not duplicate
Samples_X = X_scaled[indices]
Samples_X_df = pd.DataFrame(Samples_X, columns=X.columns)

# 5. Step 5: Make predictions with the simple model
# predictions_lr = model_lr.predict(Samples_X)
# logger.info("Predictions for Samples_X (Linear Regression Model):", predictions_lr)
# predictions_rf = model_rf.predict(Samples_X)
# logger.info("Predictions for Samples_X (Random Forest Regressor Model):", predictions_rf)
# predictions_svr = model_svr.predict(Samples_X)
# logger.info("Predictions for Samples_X (Support Vector Regressor Model):", predictions_svr)
# predictions_cnn = model_cnn.predict(Samples_X)
# logger.info("Predictions for Samples_X (CNN Model):", predictions_cnn)


# # 6. Step 6: Use ADORE to explain the predictions
adore_lr = ADORE(model=model_lr, data=Samples_X)
results_lr = adore_lr.explain()

adore_rf = ADORE(model=model_rf, data=Samples_X)
results_rf = adore_rf.explain()

adore_svr = ADORE(model=model_svr, data=Samples_X)
results_svr = adore_svr.explain()

adore_cnn = ADORE(model=model_cnn, data=Samples_X)
results_cnn = adore_cnn.explain()

# # 7. Step 7: Visualize the explanation results
# # 1. Feature
plot_feature_contributions(results_lr[0], X.columns.tolist(), "Linear Regression")
plot_feature_contributions(results_rf[0], X.columns.tolist(), "Random Forest Regressor")
plot_feature_contributions(results_svr[0], X.columns.tolist(), "Support Vector Regressor")
plot_feature_contributions(results_cnn[0], X.columns.tolist(), "CNN")
#
# # 2. Sample
# plot_sample_contributions(results_lr[0], X.columns.tolist(), "Linear Regression")
# plot_sample_contributions(results_rf[0], X.columns.tolist(), "Random Forest Regressor")
# plot_sample_contributions(results_svr[0], X.columns.tolist(), "Support Vector Regressor")
# plot_sample_contributions(results_cnn[0], X.columns.tolist(), "CNN")
#
# # 3. Heatmap
# plot_contributions_heatmap(results_lr[0], X.columns.tolist(), "Linear Regression", num_samples=10)
# plot_contributions_heatmap(results_rf[0], X.columns.tolist(), "Random Forest Regressor", num_samples=10)
# plot_contributions_heatmap(results_svr[0], X.columns.tolist(), "Support Vector Regressor", num_samples=10)
# plot_contributions_heatmap(results_cnn[0], X.columns.tolist(), "CNN", num_samples=10)
#
# # # 4. Compare
plot_comparative_feature_contributions(results_lr[6], results_cnn[6], X.columns.tolist(), label_a='LR',label_b='CNN', top_k=5)
contributions_list=[results_lr[6],results_rf[6], results_svr[6], results_cnn[6]]
model_labels =['LR', 'RF', 'SVR', 'CNN']
plot_radar_comparative_multiple_models(contributions_list, X.columns.tolist(), model_labels)
