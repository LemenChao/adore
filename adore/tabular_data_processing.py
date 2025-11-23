# -*- coding: utf-8 -*-
"""
Module Name: tabular_data_processing.py
Description:
    This module is designed to provide functions for efficiently calculating derivative matrices for tabular data.
    It contains functions for constructing the feature matrices of tabular data, perturbation functions, and functions for calculating first- and second-order derivatives.

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
    # >>> from adore.tabular_data_processing import construct_table_feature_matrix, compute_tabel_derivative_matrix
"""

import logging
from typing import Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from adore.utils import extract_prediction_scalar

# Set up logger
logger = logging.getLogger(__name__)


# Global caches for per-feature scales used to cap adaptive step growth.
_DERIV_SCALES_FOR_CAP = None
_DERIV_DELTA_METHOD_FOR_CAP = None


def _predict_continuous_output(model: object, X_single: np.ndarray, class_index: int = None) -> Tuple[float, int]:
    """
    Obtain a scalar, continuous-like output from a model, preferring probabilistic
    or margin-based scores for classifiers.

    Args:
        model: A fitted model supporting at least predict(...).
        X_single: Single input row of shape (1, n_features).
        class_index: Optional fixed class index to keep outputs consistent across perturbations.

    Returns:
        (scalar_output, class_index_used)

    Notes:
        - For classifiers, predict_proba(...) is preferred when available.
        - If probabilities are unavailable, decision_function(...) is used as a continuous proxy.
        - As a last resort, predict(...) is used, which may be piecewise-constant.
    """
    def _candidate_inputs(x):
        yield x
        if hasattr(x, "to_numpy"):
            try:
                yield x.to_numpy()
            except Exception as e:
                logger.debug(f"to_numpy fallback failed: {e}")

    def _call_model(method_name):
        if not hasattr(model, method_name):
            return None
        fn = getattr(model, method_name)
        last_exc = None
        for inp in _candidate_inputs(X_single):
            try:
                return fn(inp)
            except Exception as e:
                last_exc = e
                continue
        if last_exc:
            logger.debug(f"{method_name} failed on all candidate inputs. Last error: {last_exc}")
        return None

    proba = _call_model("predict_proba")
    if proba is not None and np.ndim(proba) == 2:
        idx = class_index if class_index is not None else int(np.argmax(proba, axis=1)[0])
        idx = min(idx, proba.shape[1] - 1)
        return float(proba[0, idx]), idx

    scores = _call_model("decision_function")
    if scores is not None:
        if np.ndim(scores) == 1:
            return float(scores[0]), class_index
        if scores.ndim == 2:
            idx = class_index if class_index is not None else int(np.argmax(scores, axis=1)[0])
            idx = min(idx, scores.shape[1] - 1)
            return float(scores[0, idx]), idx

    pred = _call_model("predict")
    if pred is None:
        raise RuntimeError("Model prediction failed for all attempted inputs.")
    if pred.shape == (1, 1):
        return float(pred[0, 0]), class_index
    if pred.shape == (1,):
        return float(pred[0]), class_index
    return float(np.ravel(pred)[0]), class_index


def _predict_transformed_output(model: object,
                                X_single,
                                class_index: int = None,
                                eps_p: float = 1e-12):
    """
    Produce a scalar output suitable for stable numerical differentiation.

    Behavior:
      - If predict_proba(...) exists, uses g(x)=log(p_cls(x)+eps_p) to reduce saturation,
        returning the associated probability for chain-rule back-transformation.
      - If only decision_function(...) or predict(...) exists, returns those values directly
        as a continuous-like proxy (no log-prob back-transformation).

    Returns:
      g_val: float, transformed scalar for differencing
      cls_idx: int or None, class index used for class-specific outputs
      p_val: float or None, original probability for cls_idx when using predict_proba
      is_logprob: bool, True iff g_val is log(prob)
    """
    if hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba")):
        proba = model.predict_proba(X_single)
        proba = np.asarray(proba, dtype=float)
        if proba.ndim == 2 and proba.shape[0] == 1:
            cls_idx = class_index if class_index is not None else int(np.argmax(proba, axis=1)[0])
            cls_idx = min(cls_idx, proba.shape[1] - 1)
            p_val = float(proba[0, cls_idx])
            g_val = float(np.log(p_val + eps_p))
            return g_val, cls_idx, p_val, True

    if hasattr(model, "decision_function") and callable(getattr(model, "decision_function")):
        scores = model.decision_function(X_single)
        scores = np.asarray(scores, dtype=float)
        s_val, _, _ = extract_prediction_scalar(scores, scores, scores)
        return float(s_val), class_index, None, False

    pred = model.predict(X_single)
    pred = np.asarray(pred, dtype=float)
    p_val, _, _ = extract_prediction_scalar(pred, pred, pred)
    return float(p_val), class_index, None, False


def construct_table_feature_matrix(data: Union[np.ndarray, list, tuple, pd.DataFrame]) -> np.ndarray:
    """
    Convert supported tabular data containers into a NumPy array.

    Supported types:
      - numpy.ndarray
      - list / tuple (converted via np.array)
      - pandas.DataFrame (converted via .values)

    Args:
        data: Input tabular data.

    Returns:
        np.ndarray: Dense feature matrix.

    Raises:
        TypeError: If the input container is not supported.
        ValueError: If the input is empty or conversion fails.
    """
    logger.info("Starting to construct table feature matrix")

    if data is None or len(data) == 0:
        raise ValueError("Input data cannot be empty")

    try:
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, (list, tuple)):
            return np.array(data)
        elif isinstance(data, pd.DataFrame):
            return data.values
        else:
            raise TypeError("Input data must be of type np.ndarray, list, tuple, or pd.DataFrame")
    except Exception as e:
        logger.error(f"Data conversion failed: {str(e)}")
        raise ValueError(f"Data conversion failed: {str(e)}")


def _compute_dense_delta(X: np.ndarray, delta_method: str, epsilon: float, strict: bool = True) -> np.ndarray:
    """
    Generate per-feature perturbation magnitudes for dense tabular inputs.

    delta_method:
        'std'    -> delta_j = epsilon * std_j
        'range'  -> delta_j = epsilon * range_j
        'single' -> delta_j = epsilon
        'iqr'    -> delta_j = epsilon * IQR_j
        'mad'    -> delta_j = epsilon * MAD_j

    Design:
        - Keep epsilon floor to prevent vanishing steps.
        - Treat near-zero feature scale as zero (fallback to epsilon).
        - Keep ONLY per-feature auto-max clamp to avoid overly large base steps.
        - No auto-min clamp; small base steps will be handled by adaptive step search later.
    """
    EPS_FLOOR = 1e-8
    ZERO_TOL = 1e-10
    AUTO_MAX_FRAC = 0.10  # keep max clamp only

    VALID_METHODS = ["std", "range", "single", "iqr", "mad"]

    if strict:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if delta_method not in VALID_METHODS:
            raise ValueError("delta_method must be one of 'std', 'range', 'single', 'iqr', or 'mad'")
    else:
        if epsilon <= 0:
            logger.warning(f"epsilon={epsilon} is non-positive; reset to EPS_FLOOR={EPS_FLOOR}")
        if delta_method not in VALID_METHODS:
            logger.warning(f"delta_method='{delta_method}' is invalid; fallback to 'std'")
            delta_method = "std"

    # ---- minimal lifecycle log ----
    logger.info("### Start computing delta matrix")

    epsilon = max(epsilon, EPS_FLOOR)

    n_samples, n_features = X.shape
    delta = np.zeros((n_samples, n_features), dtype=float)

    try:
        # ----- First pass: compute per-feature scales -----
        scales = np.zeros(n_features, dtype=float)
        for j in range(n_features):
            v = X[:, j]

            if delta_method == "std":
                scales[j] = np.std(v)
            elif delta_method == "range":
                scales[j] = np.max(v) - np.min(v)
            elif delta_method == "iqr":
                q75, q25 = np.percentile(v, [75, 25])
                scales[j] = q75 - q25
            elif delta_method == "mad":
                med = np.median(v)
                scales[j] = np.median(np.abs(v - med))
            elif delta_method == "single":
                scales[j] = 0.0
            else:
                scales[j] = np.std(v)

        # ---- debug-only scale stats (you asked) ----
        logger.debug(
            f"Feature scale stats -> min={scales.min():.6g}, "
            f"median={np.median(scales):.6g}, max={scales.max():.6g}"
        )

        # ----- Second pass: compute deltas with max clamp only -----
        zero_fallback = 0
        max_clipped = 0

        for j in range(n_features):
            scale_j = scales[j]

            # Base delta from formula
            if delta_method == "single":
                delta_val = epsilon
            else:
                if scale_j > ZERO_TOL:
                    delta_val = epsilon * scale_j
                else:
                    delta_val = epsilon
                    zero_fallback += 1

            # Auto-max clamp (per feature)
            if delta_method != "single" and scale_j > ZERO_TOL:
                max_delta_auto = AUTO_MAX_FRAC * scale_j
                if delta_val > max_delta_auto:
                    delta_val = max_delta_auto
                    max_clipped += 1

            delta[:, j] = delta_val

    except Exception as e:
        logger.error(f"Error occurred while computing delta matrix: {str(e)}")
        raise RuntimeError(f"Error occurred while computing delta matrix: {str(e)}")

    # ---- keep only core result at info ----
    logger.info(f"Delta range: [{delta.min():.6f}, {delta.max():.6f}]")

    # ---- diagnostics moved to debug to avoid info spam ----
    logger.debug(
        f"Diagnostics -> zero-scale fallback: {zero_fallback}/{n_features}, "
        f"max-clipped: {max_clipped}/{n_features}"
    )

    # ---- warnings remain (actionable signals) ----
    if zero_fallback > 0.3 * n_features:
        logger.warning("Many features have near-zero scale; deltas fall back to epsilon for these features.")
    if max_clipped > 0.3 * n_features:
        logger.warning("Auto max clamp triggered for many features; consider checking outliers or epsilon.")

    logger.info("### Finished computing delta matrix")
    return delta


def compute_table_derivative_matrix(model: object,
                                    X: np.ndarray,
                                    epsilon: float,
                                    delta_method: str,
                                    tau_threshold: float,
                                    n_jobs: int,
                                    feature_names: Union[None, list, Tuple] = None) -> np.ndarray:
    """
    Compute a per-sample, per-feature derivative matrix for tabular inputs.

    The routine estimates both first-order (Jacobian diagonal) and second-order
    (Hessian diagonal) derivatives via centered finite differences, then selects
    between them using a global tau criterion.

    Args:
        model: Fitted model supporting predict(...). Optional predict_proba(...).
        X: Input data of shape (n_samples, n_features).
        epsilon: Base scaling factor for perturbation magnitudes.
        delta_method: Per-feature scale method for base perturbations.
        tau_threshold: Threshold to choose between first- and second-order outputs.
        n_jobs: Number of parallel workers.
        feature_names: Optional column names for DataFrame-compatible models.

    Returns:
        np.ndarray of shape (n_samples, n_features): derivative matrix.
    """
    if not hasattr(model, 'predict'):
        raise ValueError("Model must have a predict method")
    if tau_threshold <= 0:
        raise ValueError("tau_threshold must be in range (0,inf)")

    logger.info("### Start computing d_matrix matrix for tabular data")
    try:
        n_samples, n_features = X.shape
        J_D = np.zeros((n_samples, n_features))
        H_D = np.zeros((n_samples, n_features))

        delta = _compute_dense_delta(X, delta_method, epsilon)

        # Precompute per-feature scales for capping adaptive step growth.
        global _DERIV_SCALES_FOR_CAP, _DERIV_DELTA_METHOD_FOR_CAP
        scales_for_cap = np.zeros(n_features, dtype=float)
        for j in range(n_features):
            v = X[:, j]
            if delta_method == "std":
                scales_for_cap[j] = np.std(v)
            elif delta_method == "range":
                scales_for_cap[j] = np.max(v) - np.min(v)
            elif delta_method == "iqr":
                q75, q25 = np.percentile(v, [75, 25])
                scales_for_cap[j] = q75 - q25
            elif delta_method == "mad":
                med = np.median(v)
                scales_for_cap[j] = np.median(np.abs(v - med))
            elif delta_method == "single":
                scales_for_cap[j] = 0.0
            else:
                scales_for_cap[j] = np.std(v)

        _DERIV_SCALES_FOR_CAP = scales_for_cap
        _DERIV_DELTA_METHOD_FOR_CAP = delta_method

        logger.info("Starting parallel computation for derivatives")
        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_derivatives)(model, X, delta, i, j, feature_names)
            for i in range(n_samples)
            for j in range(n_features)
        )

        for i, j, J_val, H_val in results:
            J_D[i, j] = J_val
            H_D[i, j] = H_val

        norm_H_D = np.linalg.norm(H_D, 'fro')
        norm_J_D = np.linalg.norm(J_D, 'fro')
        tau = norm_H_D / (1 + norm_J_D)
        logger.info(f"Calculated tau: {tau}")

        D = J_D if tau < tau_threshold else H_D
        logger.info(f"Using {'first' if tau < tau_threshold else 'second'} order derivatives")

    except Exception as e:
        logger.error(f"Error occurred while computing d_matrix matrix: {str(e)}")
        raise RuntimeError(f"Error occurred while computing d_matrix matrix: {str(e)}")

    logger.info("### Finished computing d_matrix matrix for tabular data")
    return D


def _compute_derivatives(model: object,
                         X: np.ndarray,
                         delta: np.ndarray,
                         i: int,
                         j: int,
                         feature_names: Union[None, list, Tuple]):
    """
    Estimate first- and second-order derivatives for a single sample-feature pair
    using centered finite differences with adaptive step-size enlargement.

    Returns:
        (i, j, J_val, H_val)
    """
    try:
        x_i = X[i].copy()
        base_delta = float(delta[i, j]) if delta[i, j] != 0 else 1e-6

        max_scale_mult = 32.0
        grow = 2.0
        max_tries = 8
        atol = 1e-6
        rtol = 1e-3
        eps_p = 1e-12
        ADAPT_MAX_FRAC = 0.5

        def _wrap_inputs(x_vec, d_val, force_numpy: bool = False):
            x_pos = x_vec.copy()
            x_neg = x_vec.copy()
            x_pos[j] += d_val
            x_neg[j] -= d_val
            if feature_names is not None and not force_numpy:
                try:
                    x_i_df = pd.DataFrame([x_vec], columns=feature_names)
                    x_pos_df = pd.DataFrame([x_pos], columns=feature_names)
                    x_neg_df = pd.DataFrame([x_neg], columns=feature_names)
                    return x_i_df, x_pos_df, x_neg_df
                except Exception as e:
                    logger.warning(f"Failed to reattach feature names for prediction: {e}")
            return (
                x_vec.reshape(1, -1),
                x_pos.reshape(1, -1),
                x_neg.reshape(1, -1),
            )

        force_numpy = False
        try:
            x_i_input, _, _ = _wrap_inputs(x_i, base_delta, force_numpy=force_numpy)
            g_i, cls_idx, p_i, is_logprob = _predict_transformed_output(
                model, x_i_input, class_index=None, eps_p=eps_p
            )
        except Exception as e:
            if feature_names is not None:
                logger.warning(f"Feature-named inputs failed for sample {i}, feature {j}: {e}; retrying with numpy.")
                force_numpy = True
                x_i_input, _, _ = _wrap_inputs(x_i, base_delta, force_numpy=force_numpy)
                g_i, cls_idx, p_i, is_logprob = _predict_transformed_output(
                    model, x_i_input, class_index=None, eps_p=eps_p
                )
            else:
                raise

        max_delta = abs(base_delta) * max_scale_mult
        global _DERIV_SCALES_FOR_CAP, _DERIV_DELTA_METHOD_FOR_CAP
        if _DERIV_SCALES_FOR_CAP is not None and _DERIV_SCALES_FOR_CAP.size > j:
            scale_j = float(_DERIV_SCALES_FOR_CAP[j])
            if scale_j > 0 and _DERIV_DELTA_METHOD_FOR_CAP != "single":
                max_delta = min(max_delta, ADAPT_MAX_FRAC * scale_j)

        target = atol + rtol * max(1.0, abs(g_i))

        delta_step = abs(base_delta)
        attempts = 0

        while attempts < max_tries:
            if delta_step > max_delta:
                delta_step = max_delta

            try:
                x_i_input, x_pos_input, x_neg_input = _wrap_inputs(x_i, delta_step, force_numpy=force_numpy)

                g_pos, _, _, _ = _predict_transformed_output(
                    model, x_pos_input, class_index=cls_idx, eps_p=eps_p
                )
                g_neg, _, _, _ = _predict_transformed_output(
                    model, x_neg_input, class_index=cls_idx, eps_p=eps_p
                )
            except Exception as e:
                if not force_numpy and feature_names is not None:
                    logger.warning(f"Feature-named inputs failed in loop for sample {i}, feature {j}: {e}; retrying with numpy.")
                    force_numpy = True
                    continue
                raise

            diff = max(abs(g_pos - g_i), abs(g_neg - g_i), abs(g_pos - g_neg))

            if diff > target or delta_step >= max_delta:
                if diff <= target and delta_step >= max_delta:
                    logger.debug(f"Max delta reached without measurable change for sample {i}, feature {j}")
                break

            delta_step *= grow
            attempts += 1
            logger.debug(f"Increasing delta for sample {i}, feature {j} to {delta_step} (attempt {attempts})")

        J_g = (g_pos - g_neg) / (2.0 * delta_step)
        H_g = (g_pos - 2.0 * g_i + g_neg) / (delta_step ** 2)

        if is_logprob and p_i is not None:
            J_val = p_i * J_g
            H_val = p_i * (H_g + J_g ** 2)
        else:
            J_val = J_g
            H_val = H_g

        return i, j, float(J_val), float(H_val)

    except Exception as e:
        logger.error(f"Error occurred while computing derivatives for sample {i}, feature {j}: {str(e)}")
        raise RuntimeError(f"Error occurred while computing derivatives: {str(e)}")
