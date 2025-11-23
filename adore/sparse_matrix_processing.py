# -*- coding: utf-8 -*-
"""
Module Name: sparse_matrix_processing.py
Description:
    This module is designed to provide functions to efficiently compute derivative matrices for sparse matrix data.
    It contains functions for computing first and second order derivatives.

Author:
    Fang Anran <fanganran97@126.com>

Maintainer:
    Fang Anran <fanganran97@126.com>

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
    # >>> from adore.sparse_matrix_processing import compute_sparse_derivative_matrix
"""
import logging

from joblib import Parallel, delayed
from scipy import sparse

from adore.utils import extract_prediction_scalar

# Get the logger for this module
logger = logging.getLogger(__name__)


def _compute_derivative(model, X, J_D, H_D, i, j):
    """
    Computes the derivative for a given non-zero element (i, j) in the sparse matrix.

    Args:
        model (object): A machine learning model implementing `predict_proba`.
        X (scipy.sparse.csr_matrix): Sparse input matrix.
        J_D (scipy.sparse.lil_matrix): Matrix to store first-order derivatives.
        H_D (scipy.sparse.lil_matrix): Matrix to store second-order derivatives.
        i (int): Row index of the non-zero element.
        j (int): Column index of the non-zero element.

    Returns:
        None: Updates `J_D` and `H_D` in-place.
    """
    x_i = X.getrow(i)

    # Determine the maximum TF-IDF value in column j
    max_tfidf_value = X[:, j].max()

    # Positive perturbation with max TF-IDF value
    x_perturbed_pos = x_i.copy()
    x_perturbed_pos[0, j] = max_tfidf_value

    # Negative perturbation setting value to zero
    x_perturbed_neg = x_i.copy()
    x_perturbed_neg[0, j] = 0

    # Calculate model predictions
    prediction = model.predict_proba(x_i)
    prediction_pos = model.predict_proba(x_perturbed_pos)
    prediction_neg = model.predict_proba(x_perturbed_neg)
    y_i, y_pos, y_neg = extract_prediction_scalar(prediction, prediction_pos, prediction_neg)

    # Compute derivatives
    if x_perturbed_pos[0, j] != x_i[0, j]:
        J_D[i, j] = (y_pos - y_i) / (x_perturbed_pos[0, j] - x_i[0, j])
    else:
        J_D[i, j] = 0

    H_D[i, j] = (y_pos - 2 * y_i + y_neg) / (max_tfidf_value ** 2) if max_tfidf_value != 0 else 0


def compute_sparse_derivative_matrix(model, X, tau_threshold=0.1, n_jobs=1):
    """
    Computes the derivative matrix d_matrix for sparse matrix inputs using either first-order or second-order derivatives.

    Args:
        model (object): A machine learning model (e.g., classifier) that implements `predict_proba`. The model
                         is used to generate predictions on the perturbed feature matrix.
        X (scipy.sparse.csr_matrix): Sparse matrix of shape (m, n), where `m` is the number of samples and
                                      `n` is the number of features. This matrix represents the input data
                                      for which derivatives are computed.
        tau_threshold (float, optional): A threshold value used to decide whether to use first-order (Jacobian)
                                         or second-order (Hessian) derivatives. Default is 0.1.
        n_jobs (int, optional): The number of CPU cores to use for parallel computation. Default is 1 (no parallelism).

    Returns:
        scipy.sparse.csr_matrix: The derivative matrix `d_matrix` in compressed sparse row (CSR) format. The matrix
                                  is of shape (m, n) where `m` is the number of samples and `n` is the number
                                  of features, and it contains the computed derivatives (either first or
                                  second-order) based on the perturbations applied to the feature matrix `X`.
    """
    logger.info("### Start calculating the d_matrix matrix for sparse X")

    n_samples, n_features = X.shape
    J_D = sparse.lil_matrix((n_samples, n_features))  # Initialize as LIL for efficiency
    H_D = sparse.lil_matrix((n_samples, n_features))  # Initialize as LIL for efficiency

    # Get indices of non-zero entries to iterate over
    non_zero_indices = X.nonzero()

    # Parallel computation of derivatives only for non-zero entries
    Parallel(n_jobs=n_jobs)(
        delayed(_compute_derivative)(model, X, J_D, H_D, i, j) for i, j in zip(*non_zero_indices)
    )

    # Calculate tau to decide which derivative matrix to use
    norm_H_D = sparse.linalg.norm(H_D, 'fro')  # Use sparse norm calculation
    norm_J_D = sparse.linalg.norm(J_D, 'fro')
    tau = norm_H_D / (1 + norm_J_D)

    logger.info(f"Calculated tau: {tau}")

    if tau < tau_threshold:
        logger.info("Using first-order derivatives (Jacobian).")
        D = J_D
    else:
        logger.info("Using second-order derivatives (Hessian).")
        D = H_D

    logger.info("### End calculating the d_matrix matrix for sparse X")
    return D.tocsr()
