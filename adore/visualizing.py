# -*- coding: utf-8 -*-
"""
Module Name: visualizing.py
Description:
    This module provides functionality to **visualize and interpret machine learning results**, including feature and sample contributions, as well as comparisons between different algorithms.
    It contains functions/classes for **generating various types of plots tailored to different data types and scenarios, such as feature importance visualizations, algorithm contribution comparisons, and detailed sample-level analysis**.

Author:
    Chao Lemen <chaolemen@ruc.edu.cn>

Maintainer:
    Chao Lemen <chaolemen@ruc.edu.cn>

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
    #from adore.visualizing import *
"""
import logging

import cv2
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from adore.image_data_processing import construct_image_feature_matrix  # Ensure correct import path

# Obtain a logger for this module
logger = logging.getLogger(__name__)


def _resolve_title(base: str, model_name: str = "", custom_title: str = None) -> str:
    """
    Prefer a user-provided title; otherwise append model name if given, else use the base title.
    """
    if custom_title:
        return custom_title
    if model_name:
        return f"{base} - {model_name}"
    return base


def plot_feature_contributions(contributions,
                               feature_names,
                               model_name="",
                               samples_to_plot=None,
                               title=None,
                               max_samples=10):
    """
    Plot stacked feature contributions for selected samples.

    Args:
    - contributions: Feature contribution matrix (m x n)
    - feature_names: List of feature names (n-length)
    - model_name: Name of the model (default: "")
    - samples_to_plot: List of sample indices to plot (optional). If None, pick top-k by abs contribution.
    - title: Custom plot title (optional)
    - max_samples: Max number of samples to display.
                   Default=10 and will select top-10 samples by sum(abs(contribution)) if samples_to_plot is None.
                   If max_samples is None or <=0, no limit is applied.
    """
    logger.info("Starting plot_feature_contributions function.")

    try:
        # Check if feature_names length matches the number of columns in contributions
        if len(feature_names) != contributions.shape[1]:
            logger.error("Length of feature_names does not match number of columns in contributions.")
            raise ValueError("The length of feature_names must match the number of columns in contributions.")

        n_total_samples = contributions.shape[0]

        # -------------------------------
        # >>> MOD: sample selection logic
        # -------------------------------
        if samples_to_plot is None:
            # Select top-k samples by absolute contribution magnitude
            if max_samples is None or max_samples <= 0:
                samples_to_plot = list(range(n_total_samples))
                logger.debug("samples_to_plot=None and max_samples<=0; plotting all samples.")
            else:
                # Score each sample by total absolute contribution
                scores = np.sum(np.abs(contributions), axis=1)
                top_idx = np.argsort(scores)[::-1][:min(max_samples, n_total_samples)]
                samples_to_plot = top_idx.tolist()
                logger.debug(f"samples_to_plot=None; selected top-{len(samples_to_plot)} samples by abs contribution.")
        else:
            # Ensure samples_to_plot indices are within valid range
            if any(i >= n_total_samples or i < 0 for i in samples_to_plot):
                logger.error("Sample indices in samples_to_plot are out of range.")
                raise ValueError("Sample indices in samples_to_plot must be within the range of contributions rows.")

            # If max_samples is set, optionally truncate the given list
            if max_samples is not None and max_samples > 0 and len(samples_to_plot) > max_samples:
                logger.debug(
                    f"Provided samples_to_plot has {len(samples_to_plot)} entries; "
                    f"truncating to max_samples={max_samples}."
                )
                samples_to_plot = list(samples_to_plot)[:max_samples]
        # -------------------------------
        # <<< MOD
        # -------------------------------

        num_samples = len(samples_to_plot)
        selected_contributions = contributions[samples_to_plot]

        logger.info(f"Plotting {num_samples} samples for model: {model_name}")

        # Set up the figure and axes
        fig, ax = plt.subplots(figsize=(12, 7))

        bottom = np.zeros(len(feature_names))
        for k, sample_contribution in enumerate(selected_contributions):
            ax.bar(feature_names, sample_contribution, bottom=bottom,
                   label=f'Sample {samples_to_plot[k] + 1}', alpha=0.7)
            bottom += sample_contribution
            logger.debug(f"Plotted contributions for Sample {samples_to_plot[k] + 1}")

        # Set axis labels and title
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Cumulative Contribution', fontsize=12, labelpad=15)
        plot_title = _resolve_title('Feature Contributions', model_name, title)
        ax.set_title(plot_title, fontsize=16, fontweight='bold')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Adjust legend placement to be outside the plot
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), frameon=False, ncol=1, loc='upper left',
                  bbox_to_anchor=(1, 1))

        # Add explanation text below the plot and ensure it fits within the figure
        plt.figtext(0.5, 0.03,
                    "How to read: Each colored segment represents the contribution of a feature for a specific sample.",
                    ha="center", fontsize=10, color='blue')
        plt.figtext(0.5, 0.01,
                    "Conclusion: You can observe the cumulative effect of feature contributions across samples.",
                    ha="center", fontsize=10, color='green')

        # Adjust layout to fit all elements
        plt.tight_layout(pad=3.0)

        plt.show()
        logger.info("Completed plot_feature_contributions function.")

    except Exception as e:
        logger.exception("An error occurred in plot_feature_contributions.")
        raise


def plot_sample_contributions(contributions, feature_names, model_name="", samples_to_plot=None, title=None):
    """
    Plot horizontal stacked sample contributions for selected features.

    Args:
    - contributions: Sample contribution matrix (m x n)
    - feature_names: List of feature names (n-length)
    - model_name: Name of the model (default: "Linear Regression")
    - samples_to_plot: List of sample indices to plot (optional)
    - title: Custom plot title (optional)
    """
    logger.info("Starting plot_sample_contributions function.")

    try:
        if len(feature_names) != contributions.shape[1]:
            logger.error("Length of feature_names does not match number of columns in contributions.")
            raise ValueError("The length of feature_names must match the number of columns in contributions.")

        if samples_to_plot is None:
            samples_to_plot = range(contributions.shape[0])
            logger.debug("No specific samples to plot provided; plotting all samples.")
        else:
            # Ensure samples_to_plot indices are within valid range
            if any(i >= contributions.shape[0] or i < 0 for i in samples_to_plot):
                logger.error("Sample indices in samples_to_plot are out of range.")
                raise ValueError("Sample indices in samples_to_plot must be within the range of contributions rows.")
            logger.debug(f"Samples to plot: {samples_to_plot}")

        # Extract the contributions for the specified samples
        contributions_to_plot = contributions[samples_to_plot, :]
        sample_names = [f'Sample {i}' for i in samples_to_plot]

        logger.info(f"Plotting contributions for {len(samples_to_plot)} samples for model: {model_name}")

        # Set up the figure and axes
        fig, ax = plt.subplots(figsize=(12, 7))

        # Initialize the left edge of each bar
        left_positive = np.zeros(len(sample_names))
        left_negative = np.zeros(len(sample_names))

        for i, feature_contribution in enumerate(contributions_to_plot.T):
            positive_contribution = np.where(feature_contribution > 0, feature_contribution, 0)
            negative_contribution = np.where(feature_contribution < 0, feature_contribution, 0)

            ax.barh(sample_names, positive_contribution, left=left_positive,
                    label=feature_names[i], alpha=0.7)
            ax.barh(sample_names, negative_contribution, left=left_negative,
                    alpha=0.7)

            left_positive += positive_contribution
            left_negative += negative_contribution
            logger.debug(f"Plotted feature '{feature_names[i]}' contributions.")

        # Set axis labels and title
        ax.set_xlabel('Cumulative Contribution', fontsize=12, labelpad=10)
        ax.set_ylabel('Samples', fontsize=12)
        plot_title = _resolve_title('Sample Contributions', model_name, title)
        ax.set_title(plot_title, fontsize=16, fontweight='bold')

        # Adjust legend placement to be outside the plot
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), frameon=False, ncol=1, loc='upper left',
                  bbox_to_anchor=(1, 1))  # Adjust legend position and number of columns

        # Add explanation text below the plot and ensure it fits within the figure
        plt.figtext(0.5, 0.1,
                    "How to read: Each colored segment represents the contribution of a sample for a specific feature.",
                    ha="center", fontsize=10, color='blue')
        plt.figtext(0.5, 0.08,
                    "Conclusion: You can observe the cumulative effect of sample contributions across features.",
                    ha="center", fontsize=10, color='green')

        # Adjust layout to fit all elements
        plt.tight_layout(pad=3.0)  # Adjust rect to ensure text fits within the figure area
        plt.subplots_adjust(bottom=0.2)

        # # Save figure
        # save_path = f'./results/fig/{model_name}-sample_contributions.png'
        # fig.savefig(save_path, dpi=300, bbox_inches='tight')
        # logger.info(f"Sample contributions plot saved to {save_path}")

        # Show the plot
        plt.show()
        logger.info("Completed plot_sample_contributions function.")

    except Exception as e:
        logger.exception("An error occurred in plot_sample_contributions.")
        raise


def plot_comparative_feature_contributions(f_acm_a, f_acm_b, feature_names, label_a='Algorithm A',
                                           label_b='Algorithm B',
                                           top_k=5, title=None, model_name=""):
    """
    Plot a comparative bar chart of feature contributions for two algorithms,
    focusing on the top-K most influential features from both.

    Args:
    - f_acm_a: Contribution values of features from Algorithm A (list or array).
    - f_acm_b: Contribution values of features from Algorithm B (list or array).
    - feature_names: List of feature names corresponding to the contribution values.
    - label_a: Label for Algorithm A (default: 'Algorithm A').
    - label_b: Label for Algorithm B (default: 'Algorithm B').
    - top_k: Number of top features to consider based on absolute contribution values (default: 5).
    - title: Custom plot title (optional)
    - model_name: Optional model name to append to the title.
    """
    logger.info("Starting plot_comparative_feature_contributions function.")

    try:
        # Validate input lengths
        if len(f_acm_a) != len(f_acm_b) or len(f_acm_a) != len(feature_names):
            logger.error("Lengths of f_acm_a, f_acm_b, and feature_names must be the same.")
            raise ValueError("The length of f_acm_a, f_acm_b, and feature_names must be the same.")

        # Identify the top-K features by absolute contributions for each algorithm
        top_k_indices_a = np.argsort(np.abs(f_acm_a))[-top_k:][::-1]
        top_k_indices_b = np.argsort(np.abs(f_acm_b))[-top_k:][::-1]

        # Combine top-K indices from both algorithms and ensure uniqueness
        unique_top_indices = list(set(top_k_indices_a).union(set(top_k_indices_b)))
        logger.debug(f"Unique top feature indices: {unique_top_indices}")

        # Extract the corresponding feature names and contributions for the selected features
        top_features = [feature_names[i] for i in unique_top_indices]
        top_contributions_a = [f_acm_a[i] for i in unique_top_indices]
        top_contributions_b = [f_acm_b[i] for i in unique_top_indices]

        # Set up bar chart positions
        bar_width = 0.35
        index = np.arange(len(unique_top_indices))

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars_a = ax.bar(index, top_contributions_a, bar_width, label=label_a)
        bars_b = ax.bar(index + bar_width, top_contributions_b, bar_width, label=label_b)

        logger.info("Plotted bars for both algorithms.")

        # Add labels, title, and legend
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Contribution', fontsize=12)
        plot_title = _resolve_title('Comparative Feature Contributions (Top Features from Both Algorithms)',
                                    model_name, title)
        ax.set_title(plot_title, fontsize=14, fontweight='bold')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(top_features, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=10)

        # Optimize layout for readability
        plt.tight_layout()

        # Show the plot
        plt.show()
        logger.info("Completed plot_comparative_feature_contributions function.")

    except Exception as e:
        logger.exception("An error occurred in plot_comparative_feature_contributions.")
        raise


def plot_radar_comparative_multiple_models(contributions_list, feature_names, model_labels, title=None, model_name=""):
    """
    Plot a comparative radar chart for multiple models.

    Args:
    - contributions_list: List of feature contributions for each model. Each model's contributions should be a list of numerical values.
    - feature_names: List of feature names corresponding to the contributions.
    - model_labels: Labels for each model (e.g., model names).
    - title: Custom plot title (optional)
    - model_name: Optional model name to append to the title.
    """
    logger.info("Starting plot_radar_comparative_multiple_models function.")

    try:
        # Number of features to plot
        num_features = len(feature_names)

        # Generate angles for the radar chart
        angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
        angles += angles[:1]  # Close the radar chart by appending the first angle

        # Initialize the radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Plot contributions for each model
        for i, model_contributions in enumerate(contributions_list):
            # Append the first value to the end to close the chart loop
            values = np.append(model_contributions, model_contributions[0])

            # Plot the data
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=model_labels[i])
            # Fill the area under the curve
            ax.fill(angles, values, alpha=0.25)
            logger.debug(f"Plotted radar for model: {model_labels[i]}")

        # Add feature labels to the axes
        ax.set_xticks(angles[:-1])  # Omit the duplicated closing angle
        ax.set_xticklabels(feature_names, fontsize=10)

        # Add a title to the radar chart
        plot_title = _resolve_title('Comparison of Feature Contributions Across Models', model_name, title)
        ax.set_title(plot_title, va='bottom', fontsize=16, fontweight='bold')

        # Add a legend
        ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

        # # Save figure
        # save_path = f'./results/fig/radar_comparative_multiple_models.png'
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # logger.info(f"Radar comparative plot saved to {save_path}")

        # Show the radar chart
        plt.show()
        logger.info("Completed plot_radar_comparative_multiple_models function.")

    except Exception as e:
        logger.exception("An error occurred in plot_radar_comparative_multiple_models.")
        raise


def plot_contributions_heatmap(feature_contributions, feature_names, model_name="", num_samples=10, title=None):
    """
    Plot a heatmap for feature contributions with improved aesthetics.

    Args:
    - feature_contributions: Feature contribution matrix (for tabular data: m x n, for image data: m x height x width x channels)
    - feature_names: List of feature names (for tabular data: feature names, for image data: sample names)
    - num_samples: Number of samples to plot (default: 10)
    - title: Custom plot title (optional)
    """
    logger.info("Starting plot_contributions_heatmap function.")

    try:
        # Limit the number of samples to plot
        num_samples = min(num_samples, len(feature_contributions))
        logger.debug(f"Number of samples to plot: {num_samples}")

        # Set the style and context for the plot
        sns.set(style="whitegrid", context="notebook")

        # Create a figure with a specific size
        plt.figure(figsize=(12, 8))

        # Plot the heatmap with additional formatting options
        heatmap = sns.heatmap(
            feature_contributions[:num_samples],
            annot=True,
            fmt=".2f",
            xticklabels=feature_names,
            cmap='coolwarm',
            cbar_kws={"shrink": 0.8, "label": "Contribution"},
            linewidths=0.5,
            linecolor='gray'
        )
        logger.info("Heatmap created successfully.")

        # Improve the layout and appearance
        plot_title = _resolve_title('Feature Contributions Heatmap', model_name, title)
        heatmap.set_title(plot_title, fontsize=16, fontweight='bold')
        heatmap.set_xlabel('Features', fontsize=14, fontweight='bold')
        heatmap.set_ylabel('Samples', fontsize=14, fontweight='bold', labelpad=15)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Adjust the layout to make room for rotated x-axis labels
        plt.tight_layout(pad=3)

        # # Save figure
        # save_path = f'./results/fig/{model_name}-contributions_heatmap.png'
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # logger.info(f"Contributions heatmap saved to {save_path}")

        # Show the heatmap
        plt.show()
        logger.info("Completed plot_contributions_heatmap function.")

    except Exception as e:
        logger.exception("An error occurred in plot_contributions_heatmap.")
        raise


def plot_violin_chart(feature_contributions, feature_names, model_name="", title=None):
    """
    Plot violin plot to visualize the distribution of feature contributions.

    Args:
    - feature_contributions: Feature contribution matrix (m x n)
    - feature_names: List of feature names (n-length)
    - model_name: Optional model name to append to the title.
    - title: Custom plot title (optional)
    """
    logger.info("Starting plot_violin_chart function.")

    try:
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=feature_contributions, inner='quartile')
        plt.xticks(np.arange(len(feature_names)), feature_names, rotation=45, ha='right')
        plot_title = _resolve_title('Violin Plot of Feature Contributions', model_name, title)
        plt.title(plot_title, fontsize=14, fontweight='bold')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Contribution', fontsize=12)

        logger.info("Violin plot created successfully.")

        # Adjust layout to make room for rotated x-axis labels
        plt.tight_layout(pad=3)

        # # Save figure
        # save_path = f'./results/fig/violin_chart_feature_contributions.png'
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # logger.info(f"Violin chart saved to {save_path}")

        # Add explanation text below the plot
        plt.figtext(0.5, 0.03,
                    "How to read: Each violin represents the distribution of contributions for a feature.",
                    ha="center", fontsize=10, color='blue')
        plt.figtext(0.5, 0.01,
                    "Conclusion: You can observe how feature contributions are distributed across samples.",
                    ha="center", fontsize=10, color='green')

        # Show the plot
        plt.show()
        logger.info("Completed plot_violin_chart function.")

    except Exception as e:
        logger.exception("An error occurred in plot_violin_chart.")
        raise


def plot_radar_chart(feature_contributions, feature_names, model_name="", title=None):
    """
    Plot radar chart to visualize feature contributions for each sample.

    Args:
    - feature_contributions: Feature contribution matrix (m x n)
    - feature_names: List of feature names (n-length)
    - model_name: Optional model name to append to the title.
    - title: Custom plot title (optional)
    """
    logger.info("Starting plot_radar_chart function.")

    try:
        angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 8), subplot_kw=dict(polar=True))

        for i, contribution in enumerate(feature_contributions[:5]):
            data = contribution.tolist()
            data += data[:1]
            ax.plot(angles, data, label=f'Sample {i + 1}')
            ax.fill(angles, data, alpha=0.25)
            logger.debug(f"Plotted radar for Sample {i + 1}")

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names, fontsize=10)
        plot_title = _resolve_title('Feature Importance Radar Chart', model_name, title)
        plt.title(plot_title, fontsize=14, fontweight='bold')

        # Adjust the bottom margin
        plt.subplots_adjust(bottom=0.3)

        # Add explanation text below the plot
        plt.figtext(0.5, 0.25,
                    "How to read: Each axis represents a feature, and the size of the shaded area shows the feature's contribution.",
                    ha="center", fontsize=10, color='blue')
        plt.figtext(0.5, 0.23,
                    "Conclusion: Larger shaded areas indicate higher contributions across multiple features.",
                    ha="center", fontsize=10, color='green')

        # # Save figure
        # save_path = f'./results/fig/radar_chart_feature_contributions.png'
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # logger.info(f"Radar chart saved to {save_path}")
        #
        # plt.show()
        # logger.info("Completed plot_radar_chart function.")

    except Exception as e:
        logger.exception("An error occurred in plot_radar_chart.")
        raise


def plot_bubble_chart(feature_contributions, X, feature_names, size_factor=100, title=None, model_name=""):
    """
    Plot bubble chart to visualize feature values against feature contributions.

    Args:
    - feature_contributions: Feature contribution matrix (m x n)
    - X: Feature values matrix (m x n)
    - feature_names: List of feature names (n-length)
    - size_factor (float, optional): A global scaling factor for bubble sizes. Defaults to 100.
    - title: Custom plot title (optional)
    - model_name: Optional model name to append to the title.
    """
    logger.info("Starting plot_bubble_chart function.")

    try:
        plt.figure(figsize=(10, 6))

        # Handle NaN and infinite values
        feature_contributions = np.nan_to_num(feature_contributions, nan=0.0, posinf=0.0, neginf=0.0)
        logger.debug("Handled NaN and infinite values in feature_contributions.")

        if isinstance(X, pd.DataFrame):
            X = X.values
            logger.debug("Converted X from DataFrame to NumPy array.")

        num_features = min(len(feature_names), X.shape[1])

        # We will collect custom legend patches here
        patches = []

        for i in range(num_features):
            # Bubble sizes: linear scaling based on absolute contribution
            bubble_sizes = np.abs(feature_contributions[:, i]) * size_factor

            # Draw scatter points with the next default color in the cycle
            scatter_handle = plt.scatter(
                X[:, i],
                feature_contributions[:, i],
                s=bubble_sizes,
                alpha=0.5
            )
            logger.debug(
                f"Plotted feature '{feature_names[i]}' with default color cycle and size_factor={size_factor}.")

            # Get the facecolor that Matplotlib automatically assigned
            # The scatter might have many points, so get_facecolors() is an array;
            # we can just take the first one.
            facecolor = scatter_handle.get_facecolors()[0]

            # Create a patch for legend that uses the same color, but no size info
            patch = mpatches.Patch(color=facecolor, label=f'Feature {feature_names[i]}')
            patches.append(patch)

        plot_title = _resolve_title('Bubble Chart: Feature Values vs Contributions', model_name, title)
        plt.title(plot_title, fontsize=14, fontweight='bold')
        plt.xlabel('Feature Value', fontsize=12)
        plt.ylabel('Contribution', fontsize=12)
        # Add a custom legend that shows only colors
        plt.legend(handles=patches, loc='best')

        # Adjust layout to make room for rotated x-axis labels
        plt.tight_layout()
        logger.info("Bubble chart created successfully.")

        # # Save figure
        # save_path = f'./results/fig/bubble_chart_feature_contributions.png'
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # logger.info(f"Bubble chart saved to {save_path}")

        # Show the plot
        plt.show()
        logger.info("Completed plot_bubble_chart function.")

    except Exception as e:
        logger.exception("An error occurred in plot_bubble_chart.")
        raise


def plot_explained_variance_curve(singular_values, title=None, model_name=""):
    """
    Plot cumulative explained variance curve based on singular values.

    Args:
    - singular_values: List or array of singular values.
    - title: Custom plot title (optional)
    - model_name: Optional model name to append to the title.
    """
    logger.info("Starting plot_explained_variance_curve function.")

    try:
        explained_variance = np.cumsum(singular_values) / np.sum(singular_values)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(singular_values) + 1), explained_variance, marker='o')
        plot_title = _resolve_title('Cumulative Explained Variance Curve', model_name, title)
        plt.title(plot_title, fontsize=14, fontweight='bold')
        plt.xlabel('Number of Singular Values', fontsize=12)
        plt.ylabel('Explained Variance', fontsize=12)

        # Add explanation text below the plot (3 cm)
        plt.figtext(0.5, 0.03,
                    "How to read: This curve shows how much variance is explained by the number of singular values.",
                    ha="center", fontsize=10, color='blue')
        plt.figtext(0.5, 0.01,
                    "Conclusion: You can determine the number of singular values needed to capture most of the variance.",
                    ha="center", fontsize=10, color='green')

        # Adjust layout to make room for rotated x-axis labels
        plt.tight_layout(pad=3)

        # # Save figure
        # save_path = f'./results/fig/cumulative_explained_variance_curve.png'
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # logger.info(f"Cumulative explained variance curve saved to {save_path}")

        # Show the plot
        plt.show()
        logger.info("Completed plot_explained_variance_curve function.")

    except Exception as e:
        logger.exception("An error occurred in plot_explained_variance_curve.")
        raise


def plot_text_contributions(feature_contributions, feature_names, label='Model', top_k=5, title=None, model_name=""):
    """
    Visualizes the top positive and negative contributions from features using a horizontal bar chart.

    Args:
    - feature_contributions: A 1D NumPy array of contributions for each feature.
    - feature_names: A list of feature names corresponding to the contributions.
    - label: The label for the model or feature set (default is 'Model').
    - top_k: The number of top features to display (default is 5).
    - title: Custom plot title (optional)
    - model_name: Optional model name to append to the title (defaults to label if not provided).
    """
    logger.info("Starting plot_text_contributions function.")

    try:
        if len(feature_contributions.shape) > 1:
            logger.error("Expected a 1D feature_contributions array, got more dimensions.")
            raise ValueError("Expected a 1D feature_contributions array, got more dimensions.")

        # Separate positive and negative feature_contributions
        positive_indices = np.argsort(feature_contributions)[-top_k:]
        negative_indices = np.argsort(feature_contributions)[:top_k]

        # Combine and sort indices to maintain order in the plot
        top_indices = np.concatenate((positive_indices, negative_indices))
        logger.debug(f"Top indices for plotting: {top_indices}")

        # Extract feature names corresponding to the top feature_contributions
        top_features = [feature_names[i] for i in top_indices]
        top_contributions = feature_contributions[top_indices]

        # Determine colors based on the sign of feature_contributions
        colors = ['#b57979' if c < 0 else '#576fa0' for c in top_contributions]

        # Create a plot
        plt.figure(figsize=(10, 6))

        # Bar chart for the single model
        plt.barh(top_features, top_contributions, alpha=0.7, color=colors, height=0.35)
        plt.gca().invert_yaxis()
        plt.xlabel('Contribution', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        name_for_title = model_name if model_name else label
        plot_title = _resolve_title(f'Top {top_k} Positive and Negative Contributions', name_for_title, title)
        plt.title(plot_title, fontsize=14, fontweight='bold')

        # Add a legend to distinguish positive and negative feature_contributions
        legend_elements = [
            Patch(facecolor='#b57979', label='Negative Contribution'),
            Patch(facecolor='#576fa0', label='Positive Contribution')
        ]
        plt.legend(handles=legend_elements, frameon=False)

        logger.info("Text contributions bar chart created successfully.")

        # Adjust layout to make room for rotated x-axis labels
        plt.tight_layout()

        # # Save figure
        # save_path = f'./results/fig/text_contributions_{label}.png'
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # logger.info(f"Text contributions plot saved to {save_path}")

        # Show the plot
        plt.show()
        logger.info("Completed plot_text_contributions function.")

    except Exception as e:
        logger.exception("An error occurred in plot_text_contributions.")
        raise


def visualize_feature_maps(X_data, f_acm, top_features=5, num_samples=5):
    """
    Visualize multiple images and their top N feature maps in a grid layout.
    Each row shows one sample with its feature maps.

    Args:
    - X_data: Input image data
    - f_acm: Feature contribution matrix
    - top_features: Number of top features to display per image (default: 5)
    - num_samples: Number of samples to display (default: 5)
    """
    logger.info("Starting visualize_feature_maps function.")

    try:
        # Get feature matrix
        feature_matrix, original_sizes = construct_image_feature_matrix(X_data)
        logger.debug("Constructed image feature matrix.")

        # Get indices of top N most important features (sorted descending)
        top_feature_indices = np.argsort(f_acm)[-top_features:][::-1]
        logger.debug(f"Top feature indices: {top_feature_indices}")

        # Create figure with num_samples rows, each showing original + top features
        plt.figure(figsize=(3 * (top_features + 1), 3 * num_samples))
        gs = gridspec.GridSpec(num_samples, top_features + 1)

        # Process specified number of samples
        for i in range(min(num_samples, len(X_data))):
            # Normalize the original image
            normalized_image = (X_data[i] - X_data[i].min()) / (X_data[i].max() - X_data[i].min() + 1e-5)

            # Show original image
            ax = plt.subplot(gs[i, 0])
            plt.imshow(normalized_image)
            if i == 0:
                plt.title('Original Image', fontsize=12)
            plt.axis('off')
            logger.debug(f"Plotted original image for Sample {i + 1}")

            # Show feature maps
            for j, feature_idx in enumerate(top_feature_indices):
                # Get feature map and resize to original dimensions
                feature_map = feature_matrix[i, feature_idx]

                # Normalize feature map to [0, 1]
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-5)

                # # Resize feature map using high-quality interpolation
                # resized_feature = cv2.resize(
                #     feature_map,
                #     (original_sizes[i][1], original_sizes[i][0]),
                #     interpolation=cv2.INTER_LANCZOS4  # High-quality interpolation for sharp results
                # )

                ax = plt.subplot(gs[i, j + 1])
                plt.imshow(feature_map, cmap='viridis')  # Use 'viridis' colormap for better visualization
                if i == 0:
                    plt.title(f'Feature #{feature_idx}\nContribution: {f_acm[feature_idx]:.4f}', fontsize=10)
                plt.axis('off')
                logger.debug(f"Plotted feature map {feature_idx} for Sample {i + 1}")

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        logger.info("Feature maps visualization created successfully.")

        # Save figure
        save_path = f'./results/fig/feature_maps_visualization.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature maps visualization saved to {save_path}")

        # Show the plot
        plt.show()
        logger.info("Completed visualize_feature_maps function.")

    except Exception as e:
        logger.exception("An error occurred in visualize_feature_maps.")
        raise


def plot_custom_contribution_summary(feature_contributions, X, feature_names, title=None, model_name=""):
    """
    Custom visualization of feature contributions similar to SHAP summary plot.

    Args:
    - feature_contributions: Feature contribution matrix (m x n)
    - X: Feature values matrix (m x n)
    - feature_names: List of feature names (n-length)
    - title: Custom plot title (optional)
    - model_name: Optional model name to append to the title.
    """
    logger.info("Starting plot_custom_contribution_summary function.")

    try:
        if isinstance(X, pd.DataFrame):
            X = X.values
            logger.debug("Converted X from DataFrame to NumPy array.")

        num_features = min(len(feature_names), X.shape[1], feature_contributions.shape[1])

        plt.figure(figsize=(10, 6))

        for i in range(num_features):
            contributions = feature_contributions[:, i]
            feature_values = X[:, i]

            plt.scatter(contributions, [feature_names[i]] * len(contributions),
                        c=feature_values, cmap='coolwarm', s=50, alpha=0.6, edgecolor='k')
            logger.debug(f"Plotted custom contribution summary for Feature {feature_names[i]}")

        plt.colorbar(label="Feature value")
        plt.xlabel('Contribution (Impact on model output)', fontsize=12)
        plot_title = _resolve_title('Custom Contribution Summary Plot', model_name, title)
        plt.title(plot_title, fontsize=14, fontweight='bold')

        # Add explanation text below the plot (3 cm)
        plt.figtext(0.5, 0.04,
                    "How to read: Each point represents a feature's contribution for a sample. The color indicates the feature's value.",
                    ha="center", fontsize=10, color='blue')
        plt.figtext(0.5, 0.01,
                    "Conclusion: This summary plot helps you understand how feature values relate to their impact.",
                    ha="center", fontsize=10, color='green')

        # Adjust layout to make room for rotated x-axis labels
        plt.tight_layout(pad=3)

        # # Save figure
        # save_path = f'./results/fig/custom_contribution_summary.png'
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # logger.info(f"Custom contribution summary plot saved to {save_path}")

        # Show the plot
        plt.show()
        logger.info("Completed plot_custom_contribution_summary function.")

    except Exception as e:
        logger.exception("An error occurred in plot_custom_contribution_summary.")
        raise


## ---------------------------------------------------------
def plot_freytag_stories(freytag_story, top_k_df, title=None, model_name=""):
    """
    Visualizes the Freytag Pyramid structure of a narrative story and annotates each node with sample numbers
    and feature names from the provided DataFrame.

    Args:
    top_k_df: A Pandas DataFrame containing the sample number and feature names. The first `top_k` entries are used for annotation.
    """
    logger.info("Starting plot_freytag_stories function.")

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define Freytag's Pyramid structure points
        x = [0, 1, 3, 4, 5, 6]
        y = [0, 1, 3, 5, 2, 2.5]

        ax.plot(x[:3], y[:3], color='blue', linewidth=3, label="Exposition to Climax")
        ax.plot(x[2:4], y[2:4], color='green', linewidth=3, label="Climax")
        ax.plot(x[3:], y[3:], color='red', linewidth=3, label="Falling Action to Denouement")

        ax.scatter(x[3], y[3], color='gold', s=200, marker='*', label="Climax")

        # Annotate each point with sample number and feature name
        for i, (xi, yi) in enumerate(zip(x, y)):
            if i < len(top_k_df):
                sample_num = top_k_df.iloc[i]['Sample Number']
                feature_name = top_k_df.iloc[i]['Feature Name']
                ax.text(xi, yi + 0.1, f"Sample {sample_num}\n{feature_name}", fontsize=10, ha='center')
                logger.debug(f"Annotated point ({xi}, {yi}) with Sample {sample_num} and Feature '{feature_name}'.")

        # Set x-axis labels
        ax.set_xticks(x)
        ax.set_xticklabels(['Exposition', 'Rising Action', 'Climax', 'Falling Action', 'Denouement', 'End'])

        # Remove y-axis labels
        ax.set_yticks([])

        # Set title
        plot_title = _resolve_title("Freytag's Pyramid Structure of the Story", model_name, title)
        ax.set_title(plot_title, fontsize=14, fontweight='bold')

        # Add legend
        ax.legend()

        # Adjust layout
        plt.tight_layout()

        # # Save figure
        # save_path = f'./results/fig/freytag_pyramid_structure.png'
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # logger.info(f"Freytag's Pyramid plot saved to {save_path}")

        # Show the plot
        plt.show()
        logger.info("Completed plot_freytag_stories function.")

    except Exception as e:
        logger.exception("An error occurred in plot_freytag_stories.")
        raise


import matplotlib.gridspec as gridspec

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_abt_story(
    abt_story,
    model_name="Linear Regression",
    title=None,
    wrap_width=150,
    max_lines_per_section=None,
    show_suptitle=False,
):
    """
    Visualize the ABT (AND, BUT, THEREFORE) story in a clean and dynamic n×1 table layout.

    Args:
        abt_story (tuple): A tuple containing three parts of the ABT story (and_part, but_part, therefore_part).
        model_name (str): Name of the model used in analysis. (default: "Linear Regression")
        title (str, optional): Custom title. Defaults to formatted model name.
        wrap_width (int, optional): Maximum characters per line before wrapping. Defaults to 120.
        max_lines_per_section (int or None, optional): Maximum lines shown per section for readability.
            Use None to show all lines. Defaults to None (no truncation).
    """
    try:
        # Unpack the ABT story
        and_part, but_part, therefore_part = abt_story

        # Combine titles and contents into rows
        rows = [
            ("AND", and_part, "blue"),
            ("BUT", but_part, "red"),
            ("THEREFORE", therefore_part, "green")
        ]

        def _wrap_and_limit(content: str):
            import textwrap

            # Split on existing line breaks, then wrap each line to wrap_width
            raw_lines = content.splitlines()
            wrapped_lines = []
            for line in raw_lines:
                # textwrap.wrap returns [] for empty lines; keep blank lines
                if line.strip() == "":
                    wrapped_lines.append("")
                else:
                    wrapped_lines.extend(textwrap.wrap(line, width=wrap_width))

            if max_lines_per_section is not None and len(wrapped_lines) > max_lines_per_section:
                wrapped_lines = wrapped_lines[:max_lines_per_section] + [
                    "... (truncated for readability; increase max_lines_per_section to show more)"
                ]
            return wrapped_lines

        # Dynamically calculate the number of rows and figure height
        total_lines = 0
        wrapped_contents = []  # Store wrapped contents
        for title, content, _ in rows:
            content_lines = _wrap_and_limit(content)
            wrapped_contents.append((title, content_lines))
            total_lines += 1  # For the title
            total_lines += len(content_lines) + 1  # Add content lines and spacing

        fig_height = max(8, total_lines * 0.45)  # Adjust height based on total lines

        # Create figure and grid layout
        fig = plt.figure(figsize=(12, fig_height))
        gs = GridSpec(total_lines, 1, figure=fig, hspace=0.8)  # Adjust space between rows

        # Populate grid with content
        current_row = 0
        for (title, content_lines), (_, _, color) in zip(wrapped_contents, rows):
            # Title row
            ax_title = fig.add_subplot(gs[current_row, 0])
            ax_title.axis("off")
            ax_title.text(
                0.5, 0.5,
                title,
                color=color,
                fontsize=18,  # Smaller font size for titles
                ha="center",
                va="center",
                weight="bold"
            )
            current_row += 1  # Move to the next row

            # Content row(s) with adjusted line spacing
            for line in content_lines:
                ax_content = fig.add_subplot(gs[current_row, 0])
                ax_content.axis("off")
                ax_content.text(
                    0.02, 0.5,  # Add left padding
                    line,
                    color="black",
                    fontsize=12,  # Smaller font size for content
                    ha="left",  # Align text to the left
                    va="center",
                )
                current_row += 1

            # Add spacing after each section
            current_row += 1

        # Add figure title (optional)
        if show_suptitle:
            suptitle = _resolve_title(
                "Explaining AI Decisions Based on ABT Story",
                model_name,
                title,
            )
            fig.suptitle(
                suptitle,
                fontsize=20,
                weight="bold",
                ha="center",
                y=0.98  # Adjust title position
            )

        # Add padding around the figure
        plt.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.05)

        # Show the plot
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("An error occurred in plot_abt_story:", str(e))
        raise
