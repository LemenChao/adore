# -*- coding: utf-8 -*-
"""
Module Name: image_data_processing.py
Description:
    This module is designed to provide functions to efficiently compute derivative matrices for image data.
    It contains functions for extracting feature maps from image data by VGG16, creating perturbation for feature matrix,
    reconstructing of image from perturbed feature matrix, and functions for computing first and second order derivatives.

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
Version: [0.1.2]

License:
    This module is licensed under GPL-3.0 . See LICENSE file for details.

Usage Example:
    # >>> from adore.image_data_processing import construct_image_feature_matrix, compute_image_derivative_matrix
"""
import logging
from threading import Lock

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model
from joblib import Parallel, delayed



from adore.utils import extract_prediction_scalar

# Configure logging
logger = logging.getLogger(__name__)


# 1. Preprocess images to ensure fixed input size (224x224)
def rescale_image(image, target_size=(224, 224)):
    """
    Rescale input image to target size and preprocess for feature extraction.

    Args:
    - image (numpy array): Input image data (numpy array)
    - target_size (tuple): Target size, default (224, 224)

    Returns:
    - img (numpy array): Preprocessed single channel image ready for feature extraction
    - original_size (tuple): Original image dimensions
    """
    logger.info("Starting to rescale image.")
    try:
        if isinstance(image, np.ndarray):
            img = image
        else:
            logger.error("Input must be image data (numpy array)")
            raise ValueError("Input must be image data (numpy array)")

        original_size = img.shape[:2]  # height and width
        img = cv2.resize(img, target_size)
        img = np.uint8(img)

        if len(img.shape) == 2 or img.shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        logger.info("Image rescaled and preprocessed successfully.")
        return img, original_size
    except Exception as e:
        logger.error(f"Error in rescale_image: {e}")
        raise


# 2. Extract features from image
logger.info("Initializing VGG16 base model for feature extraction.")
base_model = VGG16(weights='imagenet', include_top=False)
for layer in base_model.layers:
    layer.trainable = False
expl_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block1_pool').output)
logger.info("Feature extraction model initialized.")


def extract_features(img_array):
    """
    Extract features using explanation model. The default model is VGG16.

    Args:
    - img_array (numpy array): Preprocessed image

    Returns:
    - features (list): List of feature maps
    """
    logger.info("Starting feature extraction.")
    try:
        features = expl_model.predict(img_array)
        features = np.squeeze(features)
        features = np.transpose(features, (2, 0, 1))
        logger.info("Feature extraction completed successfully.")
        return [features[i] for i in range(features.shape[0])]
    except Exception as e:
        logger.error(f"Error in extract_features: {e}")
        raise


# 3. Create feature matrix for all images
def construct_image_feature_matrix(images, target_size=(224, 224)):
    """
    Extract feature matrix for all images.

    Args:
    - images (numpy array): Image data
    - target_size (tuple): Target size for rescaling

    Returns:
    - X (numpy array): Feature matrix
    - original_sizes (list): List of original image dimensions
    """
    logger.info("Starting to construct image feature matrix.")
    all_features = []
    original_sizes = []
    try:
        for idx, image in enumerate(images):
            logger.debug(f"Processing image {idx + 1}/{len(images)}.")
            img_array, original_size = rescale_image(image, target_size)
            features = extract_features(img_array)
            all_features.append(features)
            original_sizes.append(original_size)
            logger.debug(f"Image {idx + 1} processed successfully.")
        X = np.array(all_features)  # Shape: (n_images, n_features, height, width)
        return X, original_sizes
    except Exception as e:
        logger.error(f"Error in construct_image_feature_matrix: {e}")
        raise


# 4. Create perturbations for each feature column (positive and negative)
def create_column_perturbations(X, epsilon=0.001):
    """
    Create positive and negative perturbations for each feature column.

    Args:
    - X (numpy array): Original feature matrix
    - epsilon (float): Perturbation intensity coefficient

    Returns:
    - perturbed_features (list): List of perturbation pairs
    """
    logger.info("Starting to create column perturbations.")
    try:
        n_images = X.shape[0]
        n_features = X.shape[1]
        perturbed_features = []

        for col in range(n_features):
            logger.debug(f"Creating perturbations for feature column {col + 1}/{n_features}.")

            std_devs = [np.std(X[img_idx, col, :, :]) for img_idx in range(n_images)]
            avg_std_dev = np.mean(std_devs) or 1e-6

            pos_perturbed = np.copy(X)
            neg_perturbed = np.copy(X)

            delta = avg_std_dev * epsilon

            pos_perturbed[:, col, :, :] += delta
            neg_perturbed[:, col, :, :] -= delta

            perturbed_features.append((pos_perturbed, neg_perturbed, delta))
            logger.debug(f"Perturbations for feature column {col + 1} created with delta={delta}.")

        logger.info("Column perturbations created successfully.")
        return perturbed_features
    except Exception as e:
        logger.error(f"Error in create_column_perturbations: {e}")
        raise


# 5. Define reconstruction model outside loop to avoid retracing and reconstruct image from features
def create_reconstruction_model(input_shape):
    """
    Create model for reconstructing images from feature maps.

    Args:
    - input_shape: Shape of input feature maps

    Returns:
    - model: Keras model for reconstruction
    """
    logger.info("Creating reconstruction model.")
    try:
        inputs = tf.keras.Input(shape=input_shape)
        x = Conv2DTranspose(64, (3, 3), padding='same', activation='relu',
                            kernel_initializer=initializers.GlorotUniform(seed=42))(inputs)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2DTranspose(32, (3, 3), padding='same', activation='relu',
                            kernel_initializer=initializers.GlorotUniform(seed=42))(x)
        x = Conv2DTranspose(3, (3, 3), padding='same', activation='sigmoid',
                            kernel_initializer=initializers.GlorotUniform(seed=42))(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        logger.info("Reconstruction model prepared successfully.")
        return model
    except Exception as e:
        logger.error(f"Error in create_reconstruction_model: {e}")
        raise

reconstruction_model = None
reconstruction_lock = Lock()  # Add a lock to ensure thread safety


def reconstruct_from_features(feature_maps, original_sizes):
    """
    Reconstruct original image size from feature maps using deconvolution and upsampling.
    Uses global model to avoid retracing.
    """
    try:
        global reconstruction_model

        feature_maps = np.transpose(feature_maps, (0, 2, 3, 1))  # Shape: (n_images, height, width, channels)

        with reconstruction_lock:  # Ensure thread safety during model initialization
            if reconstruction_model is None:
                input_shape = (feature_maps.shape[1], feature_maps.shape[2], feature_maps.shape[3])
                reconstruction_model = create_reconstruction_model(input_shape)
                logger.debug("Reconstruction model initialized.")

        feature_maps = np.array(feature_maps, dtype=np.float32)
        reconstructed_images = reconstruction_model.predict(feature_maps)
        reconstructed_images = np.clip(reconstructed_images, 0, 1)

        final_images = []
        for i, size in enumerate(original_sizes):
            resized_image = cv2.resize(reconstructed_images[i], size)
            final_images.append(resized_image)
            logger.debug(f"Image {i + 1} reconstructed and resized to {size}.")

        reconstructed_array = np.array(final_images)

        return reconstructed_array
    except Exception as e:
        logger.error(f"Error in reconstruct_from_features: {e}")
        raise


# 6. Calculate image derivative matrix
predict_lock = Lock()


def safe_predict(model, data):
    """
    Safely performs prediction using the provided model with thread synchronization.
    """
    with predict_lock:
        return model.predict(data)


def _compute_derivatives(pred_model, images, i, j, perturbed_features, original_sizes):
    """
    Computes the first-order (Jacobian) and second-order (Hessian) approximations
    for the j-th feature of the i-th image.

    Args:
        pred_model (black-box model): A model that implements a `predict` method
            used for generating predictions.
        images (list or numpy.ndarray): A collection of images, each represented
            by a numpy array.
        i (int): Index for the i-th image.
        j (int): Index for the j-th feature.
        perturbed_features (list or dict): Structure containing positive and negative
            perturbations for each feature and the perturbation size (delta).
        original_sizes (list): List containing the original dimensions of each image.

    Returns:
        tuple: A tuple `(j_val, h_val)` corresponding to the approximate
            first-order (Jacobian) and second-order (Hessian) derivatives.

    Raises:
        Exception: If an error occurs during derivative computation, it is logged
            and re-raised.
    """
    try:
        # Retrieve positive/negative perturbations and delta for feature j
        pos_perturbed, neg_perturbed, delta = perturbed_features[j]

        # Reconstruct images from features
        pos_reconstructed = reconstruct_from_features(pos_perturbed[i:i + 1], [original_sizes[i]])[0]
        neg_reconstructed = reconstruct_from_features(neg_perturbed[i:i + 1], [original_sizes[i]])[0]

        # Prepare inputs for the model
        original_input = np.expand_dims(images[i], axis=0).astype(np.float32)
        pos_input = np.expand_dims(pos_reconstructed, axis=0).astype(np.float32)
        neg_input = np.expand_dims(neg_reconstructed, axis=0).astype(np.float32)

        # Perform model predictions
        original_pred = safe_predict(pred_model, original_input)
        pos_pred = safe_predict(pred_model, pos_input)
        neg_pred = safe_predict(pred_model, neg_input)

        # Extract scalar predictions for difference approximations
        y_i, y_pos, y_neg = extract_prediction_scalar(original_pred, pos_pred, neg_pred)

        # Apply central difference formulas for first- and second-order derivatives
        j_val = (y_pos - y_neg) / (2.0 * delta)
        h_val = (y_pos + y_neg - 2.0 * y_i) / (delta ** 2)

        logger.debug(f"Computed derivatives for image {i + 1}, feature {j + 1}: J={j_val}, H={h_val}")
        return j_val, h_val

    except Exception as e:
        logger.error(f"Error computing derivatives for image {i}, feature {j}: {e}")
        raise


def compute_image_derivative_matrix(pred_model, images, X, original_sizes, epsilon=0.001, tau_threshold=0.1,
                                    n_jobs=1):
    """
    Computes the derivative matrix for a set of images using the provided prediction model.

    Args:
        pred_model (black-box model): The prediction model used to generate predictions
            for the images. Must implement a `predict` method.
        images (list or numpy.ndarray): A collection of images for which the derivative
            matrix is to be computed. Each image is a numpy array, typically representing
            pixel data.
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        original_sizes (list): List of original image dimensions.
        epsilon (float, optional): The perturbation intensity coefficient used to adjust
            feature values. Defaults to `0.001`.
        tau_threshold (float, optional): The threshold ratio used to decide between using
            the first-order (Jacobian) or second-order (Hessian) derivative matrix. If the
            Frobenius norm of the Hessian is less than `tau_threshold` times that of the
            Jacobian, the Jacobian matrix is used; otherwise, the Hessian matrix is chosen.
            Defaults to `0.1`.
        n_jobs (int, optional): The number of parallel jobs to run for computing derivatives.
            Defaults to `1`, meaning no parallelization.

    Returns:
        numpy.ndarray: The computed gradient matrix, which is either the Jacobian or Hessian
            matrix, depending on their Frobenius norms. The shape of the matrix is
            `(n_samples, n_features)`.

    Raises:
        Exception: If any error occurs during the computation process, it is logged and
            re-raised.
    """
    logger.info("### Start calculating the derivative matrix for images.")
    try:
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # Allocate memory for storing Jacobian and Hessian results
        j_d_matrix = np.zeros((n_samples, n_features))
        h_d_matrix = np.zeros((n_samples, n_features))

        # Generate positive and negative perturbations for each feature based on epsilon
        perturbed_features = create_column_perturbations(X, epsilon)
        logger.info("Perturbations created for all feature columns.")

        # Calculate derivatives in parallel (or serially) for each (i, j)
        logger.info("Starting parallel computation of derivatives.")
        results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(_compute_derivatives)(pred_model, images, i, j, perturbed_features, original_sizes)
            for i in range(n_samples)
            for j in range(n_features)
        )

        # Populate j_d_matrix and h_d_matrix with computed derivative values
        logger.info("Assigning computed derivatives to j_d_matrix and h_d_matrix.")
        for idx, (j_val, h_val) in enumerate(results):
            i, j = divmod(idx, n_features)
            j_d_matrix[i, j] = j_val
            h_d_matrix[i, j] = h_val

        # Calculate the Frobenius norms of the Jacobian and Hessian matrices
        logger.info("Calculating norms of Jacobian and Hessian matrices.")
        jacobian_norm = np.linalg.norm(j_d_matrix, ord='fro')
        hessian_norm = np.linalg.norm(h_d_matrix, ord='fro')
        logger.info(f"Jacobian norm: {jacobian_norm:.4f}, Hessian norm: {hessian_norm:.4f}")

        # Decide whether to use the Jacobian or Hessian based on the threshold
        if hessian_norm < tau_threshold * jacobian_norm:
            gradient_matrix = j_d_matrix
            derivative_type = "first-order (Jacobian)"
        else:
            gradient_matrix = h_d_matrix
            derivative_type = "second-order (Hessian)"

        logger.info(f"Using {derivative_type} derivative matrix (tau_threshold={tau_threshold}).")
        logger.info("### Finished calculating the derivative matrix for images.")
        return gradient_matrix

    except Exception as e:
        logger.error(f"Error in compute_image_derivative_matrix: {e}")
        raise