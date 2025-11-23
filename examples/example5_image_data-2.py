# -*- coding: utf-8 -*-
"""
Module Name: example5_image_data-2.py
Description:
    This module demonstrates how to implement ADORE algorithm for black-box model explanation on any given images.
    A ResNet50 model is used as the prediction model images for classification. ADORE is then used to explain the model’s
    prediction.This example also provides a comparison of interpretability between the ADORE algorithm and LIME, Anchors, and SHAP.

Author:
    Lei Ming <leimingnick@ruc.edu.cn>

Maintainer:
    Lei Ming <leimingnick@ruc.edu.cn>

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
import logging

import cv2
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from adore import ADORE
from adore.image_data_processing import rescale_image
from adore.visualizing import visualize_feature_maps

# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing import image
# from lime import lime_image
# from skimage.segmentation import mark_boundaries
# import shap
# from alibi.explainers import AnchorImage
# from scipy.ndimage import zoom

# Set up logger
logger = logging.getLogger(__name__)

# Define image path (replace with your local image path)
img_path = "../data/image2.jpg"

# Read and preprocess image
img = cv2.imread(img_path)  # Load image in BGR format
img, _ = rescale_image(img) # rescale image to ResNet50 required input size (224, 224)
logger.info(f"Rescaled image shape: {img.shape}")

# Load pre-trained ResNet50 model
model = ResNet50(weights="imagenet")

# Perform ADORE explanation
adore = ADORE(model=model, data=img, n_jobs=-1)
results = adore.explain(k=5)

# Visualize feature maps using ADORE results
visualize_feature_maps(img, results[6], top_features=5, num_samples=5)

# # Perform LIME explanation
# def lime_explanation():
#     """
#     Perform LIME explanation on the provided image.
#     """
#     explainer = lime_image.LimeImageExplainer()
#     for i, image_array in enumerate(img):
#         explanation = explainer.explain_instance(
#             image_array,
#             model.predict,  # Model prediction function
#             hide_color=0,  # Color for superpixels turned off
#             top_labels=5,  # Explain top 5 labels
#             num_samples=1000,  # Number of samples for the local surrogate model
#         )
#         print(f"LIME explanation for sample {i + 1}")
#         temp, mask = explanation.get_image_and_mask(
#             explanation.top_labels[0],
#             positive_only=True,
#             num_features=5,
#             hide_rest=True
#         )
#         plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
#         plt.show()
#
# lime_explanation()
#
# # Perform SHAP explanation
# img2 = image.load_img(img_path, target_size=(224, 224))
# img_array = image.img_to_array(img2)
# img_array = np.expand_dims(img_array, axis=0)
# img_array = preprocess_input(img_array)
#
# def model_predict(x):
#     """
#     Model prediction function for SHAP.
#     """
#     return model(x)
#
#
# def shap_explanation():
#     """
#     Perform SHAP explanation on the provided image.
#     """
#     masker = shap.maskers.Image("inpaint_telea", shape=img_array[0].shape)
#     explainer = shap.Explainer(model_predict, masker=masker)
#     shap_values = explainer(img_array, max_evals=500, batch_size=50, outputs=shap.Explanation.argsort.flip[:1])
#     print("SHAP explanation for the sample.")
#     norm_img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-5)
#     shap.image_plot(shap_values, norm_img_array)
#
# shap_explanation()
#
# # Perform Anchor explanation
# def anchor_explanation():
#     """
#     Perform Anchor explanation on the provided image.
#     """
#     explainer = AnchorImage(
#         predictor=model.predict,
#         image_shape=img[0].shape,
#         segmentation_fn='slic'
#     )
#     for i, image_array in enumerate(img):
#         explanation = explainer.explain(image_array, threshold=0.95)
#         plt.imshow(explanation.anchor)
#         plt.show()
#
# anchor_explanation()
