# -*- coding: utf-8 -*-
"""
Module Name: contribution_calculating.py
Description:
    This module is intended to provide functions for contribution calculation.
    It contains functions for calculating the feature contribution matrix using the results of the stochastic singular value decomposition and for calculating the weighted feature contributions based on the provided feature and sample weights.

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
    # >>> from adore.contribution_calculating import compute_feature_contributions, compute_weighted_feature_contributions
"""

import logging

import numpy as np

# Get the logger for this module
logger = logging.getLogger(__name__)


def compute_feature_contributions(u_matrix_k, sigma_matrix_k, vt_matrix_k):
    """
    Compute the feature contributions matrix using the results from Randomized SVD.

    This function computes the feature contributions using the SVD decomposition results.

    Args:
    u_matrix_k (np.array): Left singular vectors matrix (samples).
    sigma_matrix_k (np.array): Singular values matrix (diagonal).
    vt_matrix_k (np.array): Right singular vectors matrix (features, transposed).

    Returns:
    np.array: Feature contributions matrix showing how each feature contributes to each sample.
    """
    logger.debug("### The compute_feature_contributions function starts")

    # sigma_matrix_k is a diagonal matrix, we need to multiply it back with u_matrix_k and vt_matrix_k
    sigma_matrix_k_diag = np.diag(sigma_matrix_k)

    # Compute feature contributions as u_matrix_k * sigma_matrix_k * vt_matrix_k
    feature_contributions_matrix = np.dot(np.dot(u_matrix_k, sigma_matrix_k_diag), vt_matrix_k)

    logger.debug("### The compute_feature_contributions function ends")
    return feature_contributions_matrix


def compute_weighted_feature_contributions(feature_contributions_matrix, feature_weights=None, sample_weights=None):
    """
    Compute weighted feature contributions based on provided feature and sample weights.

    This function computes the weighted contributions, adjusting the feature contributions matrix based
    on optional weights for both features and samples.

    Args:
    feature_contributions_matrix (np.array): The unweighted feature contributions matrix (from compute_feature_contributions).
    feature_weights (np.array): Optional feature weights (n,). If provided, scales feature contributions.
    sample_weights (np.array): Optional sample weights (m,). If provided, scales sample contributions.

    Returns:
    np.array: Weighted feature contributions matrix.
    """
    logger.info("### The compute_weighted_feature_contributions function starts")

    weighted_feature_contributions = feature_contributions_matrix.copy()  # Start with the original contributions matrix

    # Apply feature weights if provided
    if feature_weights is not None:
        weighted_feature_contributions *= feature_weights

    # Apply sample weights if provided
    if sample_weights is not None:
        weighted_feature_contributions = (weighted_feature_contributions.T * sample_weights).T  # Scale rows by sample weights

    logger.info("### The compute_weighted_feature_contributions function ends")
    return weighted_feature_contributions
