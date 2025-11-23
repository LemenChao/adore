# -*- coding: utf-8 -*-
"""
Module Name: core.py
Description:
    This module is the core implement of the ADORE algorithm to compute feature and sample
    contributions for black-box models, supporting both first and second-order derivatives.

Author:
    Chao Lemen <chaolemen@ruc.edu.cn>

Maintainer:
    Chao Lemen <chaolemen@ruc.edu.cn>
    Fang Anran <fanganran97@126.com>
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
    # >>> from adore import ADORE
"""

import logging
from typing import Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd

from adore.contribution_calculating import compute_feature_contributions
from adore.image_data_processing import compute_image_derivative_matrix, construct_image_feature_matrix
from adore.sparse_matrix_processing import compute_sparse_derivative_matrix
from adore.tabular_data_processing import compute_table_derivative_matrix, construct_table_feature_matrix
from adore.text_data_processing import compute_text_derivative_matrix, construct_text_feature_matrix
from adore.utils import detect_data_type, convert_to_csr_if_sparse

# Get the logger for this module
logger = logging.getLogger(__name__)

class ADORE:
    """
    ADORE: Adaptive Derivative Order Randomized Explanation
    Proposed by Borjigin Chaolemen, Renmin University of China
    Contact: chaolemen@ruc.edu.cn
    © All Rights Reserved

    This class implements the ADORE algorithm to compute feature and sample
    contributions for black-box models, supporting both first and second-order derivatives.
    """

    def __init__(
        self,
            model: object,
            data: np.ndarray,
            epsilon: float = 0.001,
            delta_method: str = 'std',
            tau_threshold: float = 0.96,
            n_jobs: int = 1,
            feature_weights: Optional[np.ndarray] = None,
            sample_weights: Optional[np.ndarray] = None,
            sparsity_threshold: float = 0.5,
            top_n_features: int = 50,
            target_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialization of the ADORE algorithm.

        Initializes the ADORE class with the given model and data.

        Args:
            model (object): Black-box model to explain.
            data (np.ndarray): Input feature matrix with shape (m, n).
            epsilon (float): Perturbation factor for the derivative calculations.
            delta_method (str): Method for calculating perturbations ('std', 'range', 'single', 'iqr', or 'mad').
            tau_threshold (float): Threshold to determine first-order or second-order derivatives.
            n_jobs (int): Number of CPU cores for parallel computation.
            feature_weights (Optional[np.ndarray]): Optional weights for features (n,).
            sample_weights (Optional[np.ndarray]): Optional weights for samples (m,).
            sparsity_threshold (float): Threshold to determine if a matrix should be converted to sparse format.
            top_n_features (int): The number of top features to use for text data. (default is 50).
            target_size (Tuple[int, int]): The target image size for resizing images before processing (default is (224, 224)).
        """
        logger.info("Initializing ADORE class.")
        logger.info(
            f"Parameters: epsilon={epsilon}, delta_method={delta_method}, tau_threshold={tau_threshold}, "
            f"n_jobs={n_jobs}, sparsity_threshold={sparsity_threshold}, top_n_features={top_n_features}, "
            f"target_size={target_size}"
        )

        # Input validation
        if model is None or data is None:
            logger.error("Model and data must not be None.")
            raise ValueError("Model and data must not be None.")

        if not isinstance(epsilon, (int, float)) or epsilon <= 0:
            logger.error("Invalid epsilon value provided.")
            raise ValueError("epsilon must be a positive number.")

        if not isinstance(tau_threshold, (int, float)) or tau_threshold <= 0:
            logger.error("Invalid tau_threshold value provided.")
            raise ValueError("tau_threshold must be a positive number.")

        if not isinstance(sparsity_threshold, (int, float)) or not (0 <= sparsity_threshold <= 1):
            logger.error("Invalid sparsity_threshold value provided.")
            raise ValueError("sparsity_threshold must be a number between 0 and 1 (inclusive).")

        self.model = model
        self.data = data
        self.epsilon = epsilon
        self.delta_method = delta_method
        self.tau_threshold = tau_threshold
        self.n_jobs = n_jobs
        self.feature_weights = feature_weights
        self.sample_weights = sample_weights
        self.sparsity_threshold = sparsity_threshold
        self.top_n_features = top_n_features
        self.target_size = target_size
        self.feature_names = None  # keep original feature names if available

        # Detect data type
        try:
            self.data_type = detect_data_type(self.data)
            logger.info(f"Detected data type: {self.data_type}")
        except Exception as e:
            logger.error(f"Error detecting data type: {str(e)}")
            raise

        # Construct feature matrix based on data type
        try:
            if self.data_type == 'tabular':
                # Preserve feature names when DataFrame was provided
                try:
                    import pandas as pd
                except ImportError:
                    pd = None

                if pd is not None and isinstance(self.data, pd.DataFrame):
                    self.feature_names = list(self.data.columns)

                self.X = construct_table_feature_matrix(self.data)
                logger.info("Constructed table feature matrix.")
            elif self.data_type == 'text':
                self.X, self.text_features = construct_text_feature_matrix(
                    self.data,
                    self.top_n_features
                )
                logger.info("Constructed text feature matrix.")
            elif self.data_type == 'sparse matrix':
                self.X = csr_matrix(self.data)
                logger.info("Converted data to CSR sparse matrix.")
            elif self.data_type == 'image':
                self.X, self.original_sizes = construct_image_feature_matrix(
                    self.data,
                    self.target_size
                )
                logger.info("Constructed image feature matrix.")
            else:
                logger.error(f"Unsupported data type: {self.data_type}")
                raise ValueError(f"Unsupported data type: {self.data_type}")
        except ValueError as ve:
            logger.error(f"Value error during feature matrix construction: {str(ve)}")
            raise
        except TypeError as te:
            logger.error(f"Type error during feature matrix construction: {str(te)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during feature matrix construction: {str(e)}")
            raise

    def extract_text_features(self) -> np.ndarray:
        """Extract text features."""
        if not hasattr(self, 'text_features'):
            logger.error("Text features not available - wrong data type.")
            raise AttributeError("Text features not available - wrong data type.")
        logger.info("Extracted text features.")
        return self.text_features

    def _log_feature_contributions(self, feature_contributions_matrix: np.ndarray):
        """
        Log feature contributions matrix for debugging.

        Args:
            feature_contributions_matrix (np.ndarray): Feature contributions matrix to log

        Raises:
            ValueError: If input matrix is None or has invalid dimensions
        """
        logger.info("### The _log_feature_contributions function starts")

        if feature_contributions_matrix is None:
            logger.error("Feature contributions matrix cannot be None.")
            raise ValueError("Feature contributions matrix cannot be None.")

        if not isinstance(feature_contributions_matrix, np.ndarray):
            logger.error("Feature contributions matrix must be a numpy array.")
            raise TypeError("Feature contributions matrix must be a numpy array.")

        logger.info(f"Feature contributions matrix shape: {feature_contributions_matrix.shape}")
        logger.info(f"Feature contributions matrix (first few rows): \n{feature_contributions_matrix[:5]}")

        logger.info("### The _log_feature_contributions function ends")

    def _compute_derivative_matrix(self) -> np.ndarray:
        """
        Compute the Derivative Matrix d_matrix using parallel processing for perturbations.

        Returns:
            np.ndarray or scipy.sparse.csr_matrix: Derivative matrix (dense or sparse)

        Raises:
            ValueError: If data type is unsupported or computation fails
        """
        logger.info("### The _compute_derivative_matrix function starts")
        logger.info(f"Using n_jobs: {self.n_jobs}")

        try:
            if self.data_type == 'tabular':
                d_matrix = compute_table_derivative_matrix(
                    self.model,
                    self.X,
                    self.epsilon,
                    self.delta_method,
                    self.tau_threshold,
                    self.n_jobs,
                    self.feature_names
                )
                logger.info("Computed derivative matrix for tabular data.")
            elif self.data_type == 'text':
                d_matrix = compute_text_derivative_matrix(
                    self.model,
                    self.X,
                    self.text_features,
                    self.data,
                    self.tau_threshold,
                    self.n_jobs
                )
                logger.info("Computed derivative matrix for text data.")
            elif self.data_type == 'sparse matrix':
                d_matrix = compute_sparse_derivative_matrix(
                    self.model,
                    self.X,
                    self.tau_threshold,
                    self.n_jobs
                )
                logger.info("Computed derivative matrix for sparse matrix data.")
            elif self.data_type == 'image':
                d_matrix = compute_image_derivative_matrix(
                    self.model,
                    self.data,
                    self.X,
                    self.original_sizes,
                    self.tau_threshold,
                    self.epsilon,
                    self.n_jobs
                )
                logger.info("Computed derivative matrix for image data.")
            else:
                logger.error(f"Unsupported data type: {self.data_type}")
                raise ValueError(f"Unsupported data type: {self.data_type}")

            if d_matrix is None or d_matrix.size == 0:
                logger.error("Failed to compute derivative matrix.")
                raise ValueError("Failed to compute derivative matrix")

            logger.info(
                f"Derivative matrix shape: {d_matrix.shape}, type: {'sparse' if isinstance(d_matrix, csr_matrix) else 'dense'}")
            logger.info("### The _compute_derivative_matrix function ends")
            return d_matrix

        except ValueError as ve:
            logger.error(f"Value error during derivative matrix computation: {str(ve)}")
            raise
        except TypeError as te:
            logger.error(f"Type error during derivative matrix computation: {str(te)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during derivative matrix computation: {str(e)}")
            raise

    def _explain_contributions(self, d_matrix: np.ndarray, n_jobs: int, k: Optional[int] = None) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate contribution matrix based on SVD.

        Args:
            d_matrix (np.ndarray): Derivative matrix
            n_jobs (int): Number of parallel jobs
            k (Optional[int]): Number of singular values

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Contributions matrix and SVD components
        """
        if d_matrix is None or d_matrix.size == 0:
            logger.error("Invalid derivative matrix provided to _explain_contributions.")
            raise ValueError("Invalid derivative matrix")

        logger.info("### Start getting the contribution matrix based on the SVD")
        logger.info(f"Using n_jobs: {n_jobs}")

        try:
            u_matrix_k, sigma_matrix_k, vt_matrix_k = randomized_svd(
                d_matrix,
                n_components=k if k is not None else min(d_matrix.shape),
                random_state=42
            )
            logger.info(f"Randomized SVD completed with n_components={k if k else 'min(d_matrix.shape)'}")
            logger.info(
                f"u_matrix_k shape: {u_matrix_k.shape}, sigma_matrix_k shape: {sigma_matrix_k.shape}, vt_matrix_k shape: {vt_matrix_k.shape}")

            logger.info("Starting parallel computation of feature contributions.")

            # Parallel optimization: Batch processing can be implemented here if needed
            contributions_matrix = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(compute_feature_contributions)(u_matrix_k[i], sigma_matrix_k, vt_matrix_k)
                for i in range(len(u_matrix_k))
            )

            contributions_matrix = np.array(contributions_matrix)

            logger.info("Completed parallel computation of feature contributions.")

            if contributions_matrix is None or contributions_matrix.size == 0:
                logger.error("Failed to compute contributions matrix.")
                raise ValueError("Failed to compute contributions matrix")

            logger.info("### End of SVD to get the contribution matrix")
            return contributions_matrix, u_matrix_k, sigma_matrix_k, vt_matrix_k

        except ValueError as ve:
            logger.error(f"Value error in contribution calculation: {str(ve)}")
            raise
        except TypeError as te:
            logger.error(f"Type error in contribution calculation: {str(te)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in contribution calculation: {str(e)}")
            raise

    def explain(self, k: Optional[int] = None) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Provide explanation by computing feature contributions using Randomized SVD.

        Args:
            k (Optional[int]): Number of singular values to keep (optional)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                Contributions matrix and various matrices showing contributions at different levels
        """
        logger.info("### The explain function starts")

        # Validate parameter k
        if k is not None:
            if not isinstance(k, int) or k <= 0:
                logger.error("Parameter k must be a positive integer.")
                raise ValueError("Parameter k must be a positive integer.")

        try:
            # Compute derivative matrix
            d_matrix = self._compute_derivative_matrix()
            logger.info('d_matrix is ready')

            # Convert to CSR if sparse
            d_matrix = convert_to_csr_if_sparse(d_matrix, self.sparsity_threshold)
            logger.info(f"Converted d_matrix to CSR format: {isinstance(d_matrix, csr_matrix)}")

            # Explain contributions
            contributions_matrix, u_matrix_k, sigma_matrix_k, vt_matrix_k = self._explain_contributions(
                d_matrix,
                self.n_jobs,
                k
            )

            # Compute various contribution metrics
            e_acm = contributions_matrix
            e_rcm = np.abs(e_acm) / np.sum(np.abs(e_acm))

            s_acm = np.sum(u_matrix_k, axis=1)
            s_rcm = np.abs(s_acm) / np.sum(np.abs(s_acm))

            f_acm = np.sum(vt_matrix_k, axis=0)
            f_rcm = np.abs(f_acm) / np.sum(np.abs(f_acm))

            logger.info("Computed contribution metrics.")

            # Log feature contributions
            self._log_feature_contributions(contributions_matrix)

            logger.info("### The explain function ends.")
            return contributions_matrix, d_matrix, e_acm, e_rcm, s_acm, s_rcm, f_acm, f_rcm

        except Exception as e:
            logger.error(f"Error in explanation process: {str(e)}")
            raise
