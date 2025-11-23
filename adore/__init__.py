# __init__.py

# ADORE: Adaptive Derivative Order Randomized Explanation
# Version Information
__version__ = "0.1.2"

# Log initialization (Logging configuration for the ADORE package)
import logging
import os

from .contribution_calculating import compute_feature_contributions, compute_weighted_feature_contributions
# Import key modules and classes for easy access
from .core import ADORE
from .image_data_processing import rescale_image, extract_features, construct_image_feature_matrix, \
    compute_image_derivative_matrix
from .sparse_matrix_processing import compute_sparse_derivative_matrix
from .storytelling import generate_freytag_story, generate_abt_story, select_top_k_feature_contributions
from .tabular_data_processing import construct_table_feature_matrix, compute_table_derivative_matrix
from .text_data_processing import construct_text_feature_matrix, compute_text_derivative_matrix
from .utils import detect_data_type, convert_to_csr_if_sparse, extract_prediction_scalar
from .visualizing import (plot_feature_contributions, plot_sample_contributions, plot_comparative_feature_contributions,
                          plot_radar_comparative_multiple_models, plot_contributions_heatmap, plot_radar_chart,
                          plot_bubble_chart, plot_explained_variance_curve, plot_custom_contribution_summary,
                          plot_violin_chart, plot_abt_story, plot_freytag_stories, plot_text_contributions,
                          visualize_feature_maps)


# Setting up logging to capture the package's execution flow and debug information
def setup_logging():
    # Make sure the log folder debug exists, and create it if it does not
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.DEBUG,  # Set the global log level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=os.path.join(log_dir, 'adore.log'),
        filemode='a'
    )

    # Create a logger instance
    return logging.getLogger(__name__)


# Initialize the logger
logger = setup_logging()

setup_logging()

# Explicitly expose core classes and functions for user-friendly access
__all__ = [
    "ADORE",  # Core ADORE class for model explanation
    "compute_feature_contributions", "compute_weighted_feature_contributions",  # Feature contribution functions
    "construct_table_feature_matrix", "compute_table_derivative_matrix",
    # Tabular data processing functions
    "construct_text_feature_matrix", "compute_text_derivative_matrix",  # Text data processing functions
    "rescale_image", "extract_features", "construct_image_feature_matrix", "compute_image_derivative_matrix",
    # Image data processing functions
    "compute_sparse_derivative_matrix",  # Sparse data processing functions
    "generate_abt_story", "generate_freytag_story", "select_top_k_feature_contributions",  # Storytelling functions
    "plot_feature_contributions", "plot_sample_contributions", "plot_comparative_feature_contributions",
    "plot_radar_comparative_multiple_models", "plot_contributions_heatmap", "plot_radar_chart",
    "plot_bubble_chart", "plot_explained_variance_curve", "plot_custom_contribution_summary",
    "plot_violin_chart", "plot_abt_story", "plot_freytag_stories", "plot_text_contributions",
    "visualize_feature_maps",  # Visualization functions
    "detect_data_type", "convert_to_csr_if_sparse", "extract_prediction_scalar"  # Utility functions
]

# Additional initialization code can be added below as needed
logger.info(f"ADORE package version {__version__} initialized successfully.")
