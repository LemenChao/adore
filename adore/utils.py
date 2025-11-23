# -*- coding: utf-8 -*-
"""
Module Name: utils.py
Description:
    This module aims to provide some reusable tool functions.
    It contains functions for determining the type of given data, checking the sparsity of matrices, and extracting scalar values from the model prediction output.
Author:
    Chao Lemen <chaolemen@ruc.edu.cn>

Maintainer:
    Chao Lemen <chaolemen@ruc.edu.cn>

Contributors:
    Chao Lemen <chaolemen@ruc.edu.cn>
    Lei Ming <leimingnick@ruc.edu.cn>
    Fang Anran <fanganran97@126.com>

Created on: 2024-12-25
Last Modified on: 2024-12-25
Version: [0.1.2]

License:
    This module is licensed under GPL-3.0 . See LICENSE file for details.

Usage Example:
    # >>> from adore.utils import extract_prediction_scalar
"""

import logging

import numpy as np
from scipy.sparse import issparse, csr_matrix

# Set up logger
logger = logging.getLogger(__name__)


def detect_data_type(data):
    """
    Determines the type of the given data. It checks for the following types:
    - Text data (list of strings)
    - Tabular data (list of lists/tuples, NumPy 2D arrays, or Pandas DataFrame)
    - Image data (4D arrays with RGB channels)
    - Sparse matrices
    - Unknown types

    Args:
    data: The input data to classify. It can be a Python list, NumPy array, Pandas DataFrame, or other formats.

    Returns:
    str: The type of the data. Possible values:
        - 'text'
        - 'tabular'
        - 'image'
        - 'sparse matrix'
        - 'unknown'
    """
    data_type = "unknown"  # Default to unknown type
    logger.info("Starting data type detection...")

    # **Check for text data (list of strings)**
    if isinstance(data, list) and all(isinstance(item, str) for item in data):
        data_type = "text"
        logger.info("The data type is text")

    # **Check for tabular data (list of lists or tuples)**
    elif isinstance(data, list) and all(isinstance(row, (list, tuple)) for row in data):
        data_type = "tabular"
        logger.info("The data type is tabular")

    # **Check for image data (4D nested lists with RGB channels)**
    elif isinstance(data, list) and all(isinstance(image, list) for image in data):
        # Perform detailed checks to validate 4D array structure
        first_image = data[0]
        if isinstance(first_image, list) and len(first_image) > 0:
            if isinstance(first_image[0], list) and len(first_image[0]) > 0:
                if isinstance(first_image[0][0], list) and len(first_image[0][0]) == 3:
                    # Assumes last dimension is for RGB channels
                    data_type = "image"
                    logger.info("The data type is image")

    # **Check for NumPy array**
    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            if data.ndim == 2:  # 2D array for tabular data
                data_type = "tabular"
                logger.info("The data type is tabular")
            elif data.ndim == 3 and data.shape[-1] == 3:  # 3D array for single image
                data_type = "image"
                logger.info("The data type is image")
            elif data.ndim == 4 and data.shape[-1] == 3:  # 4D array for batch of images
                data_type = "image"
                logger.info("The data type is image")
    except ImportError:
        pass  # NumPy not available

    # **Check for Pandas DataFrame**
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            data_type = "tabular"
            logger.info("The data type is tabular")
    except ImportError:
        pass  # Pandas not available

    # **Check for sparse matrix (from SciPy)**
    try:
        from scipy.sparse import issparse
        if issparse(data):
            data_type = "sparse matrix"
            logger.info("The data type is sparse matrix")
    except ImportError:
        pass  # SciPy not available

    # **Default case if no match is found**
    if data_type == "unknown":
        logger.warning("The data type is unknown")

    return data_type


# Check the sparsity of the matrix
def convert_to_csr_if_sparse(matrix, sparsity_threshold):
    """
    Checks the sparsity of a matrix and converts it to CSR format if the sparsity level exceeds the threshold.

    Args:
    matrix: The matrix to check for sparsity. This can be either a dense or sparse matrix.
    sparsity_threshold: The threshold above which the matrix will be converted to CSR format.

    Returns:
    scipy.sparse.csr_matrix: The matrix in CSR format if the sparsity level exceeds the threshold,
                              otherwise returns the original matrix.
    """
    logger.info("### The convert_to_csr_if_sparse function starts")

    if issparse(matrix):
        total_elements = matrix.shape[0] * matrix.shape[1]
        zero_elements = total_elements - matrix.nnz
    else:
        total_elements = matrix.size
        zero_elements = np.sum(matrix == 0)

    sparsity = zero_elements / total_elements
    logger.info(f"Sparsity level: {sparsity:.4f}")

    if sparsity > sparsity_threshold:
        logger.info(f"Converting matrix to CSR format due to high sparsity level: {sparsity:.4f}")
        if not issparse(matrix):
            matrix = csr_matrix(matrix)
        return matrix.tocsr()
    else:
        logger.info(f"Sparsity level {sparsity:.4f} is below the threshold, keeping current format.")
        return matrix


def extract_prediction_scalar(prediction, prediction_pos, prediction_neg):
    """
    Extract scalar values from model prediction outputs.

    Args:
    prediction: The main prediction output, which could be a scalar, a (1, 1) array,
                a 1D array, or a probability distribution array.
    prediction_pos: The positive class prediction value associated with the main prediction output.
    prediction_neg: The negative class prediction value associated with the main prediction output.

    Returns:
    A tuple containing scalar values extracted from prediction, positive_prediction,
    and negative_prediction, based on the input shapes and types.
    """
    # Log at debug to avoid verbose output in parallel loops
    logger.debug("Extracting scalar values from predictions")

    if np.isscalar(prediction):
        return prediction, prediction_pos, prediction_neg

    # If prediction is an array of shape (1, 1)
    elif prediction.shape == (1, 1):
        return prediction[0, 0], prediction_pos[0, 0], prediction_neg[0, 0]

    # If prediction is a one-dimensional array
    elif prediction.ndim == 1 and prediction.size == 1:
        return prediction[0], prediction_pos[0], prediction_neg[0]

    # If prediction is a probability distribution array (for multi-class classification)
    elif prediction.ndim == 2 and prediction.shape[0] == 1:
        # Extract the index of the maximum probability
        max_index = np.argmax(prediction[0])
        # Return the value of the maximum probability
        return prediction[0][max_index], prediction_pos[0][max_index], prediction_neg[0][max_index]

    else:
        logger.error("Unexpected prediction shape or type")
        return None, None, None
