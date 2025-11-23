# ADORE: Adaptive Derivative Order Randomized Explanation

<p align="center">
  <img src="https://raw.githubusercontent.com/LemenChao/adore/main/images/logo.svg"
       alt="Logo" width="300" />
</p>

**Author**: Lemen Chao, Anran Fang, Ming Lei, Renmin University of China  
**Contact**: chaolemen@ruc.edu.cn

ADORE (Adaptive Derivative Order Randomized Explanation) is an advanced explainability framework designed to provide robust and adaptive insights into predictions made by black-box models. The algorithm bridges the gap between first-order and second-order derivatives to provide granular, mathematically rigorous feature attributions, making it uniquely suited for interpreting both simple and complex machine learning models.

## Key Innovations and Features

### 1. Unified Framework for First- and Second-Order Explanations
ADORE introduces a unified approach combining first-order (Jacobian) and second-order (Hessian) derivatives, providing a more detailed understanding of model behaviors. While most existing explainability frameworks are restricted to first-order derivatives, ADORE's ability to incorporate second-order effects allows it to capture interactions between features, making it especially powerful for complex models.

### 2. Adaptive Perturbation Based on Data Dynamics
ADORE leverages adaptive perturbation, adjusting the granularity of perturbations based on the standard deviation or range of the data. This ensures that explanations remain sensitive to the scale and distribution of features, leading to more precise and context-aware interpretations of model decisions.

### 3. Randomized Singular Value Decomposition (SVD) for Efficient Decomposition
By incorporating randomized SVD, ADORE is able to efficiently handle large datasets and sparse matrices, making it faster and more scalable compared to traditional SVD-based explanation techniques. This optimization allows the algorithm to break down high-dimensional data into meaningful components for both local and global model interpretability.

### 4. Support for Multiple Data Types (Tabular, Text, Image)
Unlike many algorithms that are confined to a specific data type, ADORE is versatile and can process tabular, text, and image data. This flexibility makes it suitable for a wide variety of real-world machine learning tasks, from NLP to computer vision.

### 5. Dynamic Sparsity Detection
ADORE integrates dynamic sparsity detection, which automatically determines when to switch to sparse matrix representations based on the sparsity level of the data. This ensures computational resources are used efficiently, particularly in high-dimensional datasets where many features may not significantly contribute to the model’s predictions.

### 6. Feature Weighting and Sample Weighting
ADORE supports the inclusion of feature and sample weights, allowing users to customize the importance of specific features or samples during the explanation process. This makes the algorithm highly adaptable to use cases where certain features or instances are known to carry more importance.

### 7. Comprehensive Visualizations
ADORE includes a suite of visualization tools to help users interpret model explanations. These tools offer insights into feature importance, interactions, and overall model behavior through intuitive visual outputs such as bar charts, heatmaps, and comparative plots across different models.

## Why ADORE?

Existing explainability techniques like SHAP and LIME often focus on first-order effects or are limited in their ability to efficiently explain complex models. ADORE extends these capabilities by offering:

- **Higher-order explainability**: Capturing interactions and dependencies between features through second-order derivatives.
- **Adaptiveness**: Perturbations dynamically adjust based on data characteristics, providing more precise attributions.
- **Scalability**: The use of randomized SVD makes it feasible to handle large datasets with high efficiency.
- **Versatility**: Works across various data types, making it applicable to a wide range of machine learning tasks.

## Installation

### Prerequisites
- Python 3.9-3.12
- Required packages: `numpy`, `scipy`, `joblib`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `cv2`, `spacy`, `nltk`, `keybert`, `tensorflow`, `tf-keras`, `torch`, `transformers`, `pyspellchecker`, `tensorflow-datasets`

### Installing ADORE
Download the package from PyPI:

```bash
pip install adore
```

## Usage

### Tabular Data Example

To explain tabular data using ADORE:

```python
from adore import ADORE
from sklearn.ensemble import RandomForestClassifier

# Train a black-box model (e.g., RandomForest)
model = RandomForestClassifier()
model.fit(X, y)

# Explain the model
adore = ADORE(model=model, data=X)
contributions, _, _, _, _, _, _, _ = adore.explain()
```

### Text Data Example

For text data, use the `TfidfVectorizer` to vectorize the input before explaining the model:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from adore import ADORE

# Vectorize the text data
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(text_data)

# Explain the model
adore = ADORE(model=trained_model, data=X_train)
contributions = adore.explain()
```

### Image Data Example

To explain image data:

```python
from adore import ADORE

# Load image data
image_data = load_images()

# Explain the model
adore = ADORE(model=image_model, data=image_data)
contributions = adore.explain()
```

## Modules

ADORE consists of several key modules:

- **core.py**: Implements the `ADORE` class and core functionality for explaining models.
- **contribution_calculating.py**: Handles feature contribution calculations.
- **tabular_data_processing.py**: Manages tabular data explanation.
- **text_data_processing.py**: Manages text data explanation.
- **image_data_processing.py**: Provides image-based explanation capabilities.
- **storytelling.py**: Generates Freytag and ABT data stories.
- **visualizing.py**: Includes several visualization functions.

### core.py

This module implements the core algorithm, including key functions like:

- `ADORE.__init__()`: Initializes the ADORE class with essential parameters such as the model, input data, perturbation method, etc.
- `ADORE.extract_text_features()`: Extracts text features.
- `ADORE._log_feature_contributions()`: Logs the computed feature contributions.
- `ADORE._compute_derivative_matrix()`: Computes the derivative matrix with sparsity detection.
- `ADORE._explain_contributions()`: Calculates the contribution matrix using SVD.
- `ADORE.explain()`: Provides an explanation by computing feature contributions using randomized SVD.

### contribution_calculating.py

Handles feature contribution calculations with functions like:

- `compute_feature_contributions(U_k, Sigma_k, V_k_T)`: Computes the feature contribution matrix using SVD results.
- `compute_weighted_feature_contributions(feature_contributions_matrix, feature_weights=None, sample_weights=None)`: Computes weighted feature contributions based on optional feature and sample weights.


### tabular_data_processing.py

Manages tabular data explanation:

- `construct_table_feature_matrix()`: Converts input data into a NumPy array.
- `compute_dense_delta()`: Generates adaptive perturbation values for input data.
- `compute_table_derivative_matrix()`: Computes the derivative matrix for tabular data.

### text_data_processing.py

Manages text data explanation:

- `construct_text_feature_matrix()`: Constructs a feature matrix from text data using keyBERT to extract the top N keywords for each document.
- `compute_text_derivative_matrix()`: Computes the derivative matrix for text data using a model's predictions.

### image_data_processing.py

Provides image-based explanation capabilities, with functions such as:

- `rescale_image()`: Rescales input image to target size.
- `extract_features()`: Extracts features from an image using the provided explanation model (default is VGG16).
- `compute_image_derivative_matrix()`: Computes the derivative matrix for a set of images using the provided prediction model.

### storytelling.py

Generate data stories:

- `generate_freytag_story()`: Create a Freytag pyramid structured narrative story based on the feature contribution analysis.
- `generate_abt_story()`: Create an ABT (And, But, Therefore) format narrative story based on feature contribution analysis.

### visualizing.py

Includes several visualization functions:

- `plot_feature_contributions()`: Visualizes feature contributions in bar chart form.
- `plot_sample_contributions()`: Plots horizontal stacked sample contributions for selected features.
- `plot_comparative_feature_contributions()`: Plots a comparative bar chart of feature contributions for two algorithms.
- `plot_radar_comparative_multiple_models()`: Plots a comparative radar chart for multiple models.

## License

This project is licensed under the GPL-3.0 License - see the LICENSE file for details.

## Contributing to ADORE

All kinds of contributions, bug reports, bug fixes, documentation improvements, or other feedbacks are welcome.

You can contribute in the following ways:

- **Report issues or suggest improvements** by opening an [issue on GitHub](https://github.com/LemenChao/adore/issues).  
  Please include clear steps to report bugs, vital information, and any relevant details.

- **Contact us via email** at **chaolemen@ruc.edu.cn** for detailed discussions, proposals, or questions that don’t fit well into a GitHub issue.

Thank you for helping improve this project. Every suggestion and fix matters!


## Contact

For any inquiries or contributions, please contact Lemen Chao at chaolemen@ruc.edu.cn.
