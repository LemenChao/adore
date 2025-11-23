# -*- coding: utf-8 -*-
"""
Module Name: text_data_processing.py
Description:
    This module is designed to provide functions to efficiently compute derivative matrices for textual data.
    It contains functions for constructing feature matrices for text data, positive and negative perturbation functions for generating keywords, and functions for computing first and second order derivatives.

Author:
    Fang Anran <fanganran97@126.com>

Maintainer:
    Fang Anran <fanganran97@126.com>

Contributors:
    Chao Lemen <chaolemen@ruc.edu.cn>
    Lei Ming <leimingnick@ruc.edu.cn>
    Fang Anran <fanganran97@126.com>

Created on: 2024-12-25
Last Modified on: 2025-11-07
Version: [0.1.2]

License:
    This module is licensed under GPL-3.0 . See LICENSE file for details.

Usage Example:
    # >>> from adore.text_data_processing import construct_text_feature_matrix, compute_text_derivative_matrix
"""

import logging
from functools import lru_cache
from typing import List, Tuple

import numpy as np
from scipy import sparse

import spacy
from joblib import Parallel, delayed
from nltk.corpus import wordnet
from nltk.metrics import edit_distance
from sklearn.feature_extraction.text import CountVectorizer

from adore.utils import extract_prediction_scalar

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO

# Lazily load spaCy model

_nlp = None  # module-level cache

def get_nlp():
    """
    Lazily load spaCy model.
    """
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        _nlp = spacy.load("en_core_web_sm")
    except OSError as e:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Install it with:\n"
            "  python -m spacy download en_core_web_sm\n"
            "Or install a pinned wheel matching your spaCy version."
        ) from e

    return _nlp




def construct_text_feature_matrix(data: List[str], top_n_features: int, logger=None) -> Tuple[np.ndarray, List[str]]:
    """
    Constructs a feature matrix from text data using Bag of Words (word frequency counting).

    Args:
        data (List[str]): A list of text documents to be transformed.
        top_n_features (int): The number of top features (most frequent words) to select.
        logger: Optional logger instance.

    Returns:
        Tuple[np.ndarray, List[str]]: Feature matrix (shape: [n_documents, top_n_features])
                                      and feature names (selected words).
    """
    if not isinstance(top_n_features, int) or top_n_features <= 0:
        raise ValueError("top_n_features must be a positive integer.")

    if logger:
        logger.info("Starting feature construction using Bag of Words (word frequency)")


    vectorizer = CountVectorizer(
        stop_words='english',
        max_features=top_n_features,  # Only the top_n most frequent words are retained as features.
        token_pattern=r'(?u)\b\w\w+\b'  # Keep only words with at least two characters (filter out single-letter noise)
    )


    X = vectorizer.fit_transform(data).toarray()


    feature_names = vectorizer.get_feature_names_out().tolist()


    if len(feature_names) > top_n_features:
        raise ValueError(f"Finding {len(feature_names)} unique words exceeds the limit of {top_n_features} top feature words ")

    if logger:
        logger.info(f"Bag of Words feature construction completed. Selected {len(feature_names)} features.")

    return X, feature_names


@lru_cache(maxsize=1000)
def _fetch_wordnet_data(word: str) -> Tuple[set, set]:
    """
    Fetch synonyms and antonyms for a single word using WordNet.
    Args:
        word (str): Input word.
    Returns:
        Tuple[set, set]: Synonyms and antonyms for the word.
    """
    synonyms = set()
    antonyms = set()

    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
            if lemma.antonyms():
                antonyms.add(lemma.antonyms()[0].name())

    return synonyms, antonyms


def _find_synonyms_antonyms_with_spacy(phrase: str, nlp_instance, logger=None) -> Tuple[List[str], List[str]]:
    """
    Uses spaCy and WordNet to find synonyms and antonyms for a phrase.

    Args:
        phrase (str): Input phrase.
        nlp_instance: spaCy NLP model instance.
        logger: Optional logger instance.

    Returns:
        Tuple[List[str], List[str]]: Synonyms and antonyms.
    """
    if logger:
        logger.info(f"Finding synonyms and antonyms for the phrase: {phrase}")

    doc = nlp_instance(phrase)
    synonyms = set()
    antonyms = set()

    for token in doc:
        token_synonyms, token_antonyms = _fetch_wordnet_data(token.text)
        synonyms.update(token_synonyms)
        antonyms.update(token_antonyms)

    if logger:
        logger.info(f"Found {len(synonyms)} synonyms and {len(antonyms)} antonyms.")

    return list(synonyms), list(antonyms)


def _compute_word_distance(word1, word2, offset=0.1):
    """
    Computes the Levenshtein distance between two words with a small constant offset.

    Args:
    - word1 (str): The first word to compare.
    - word2 (str): The second word to compare.

    Returns:
    - float: The Levenshtein distance with a small offset added.
    """
    if not isinstance(word1, str) or not isinstance(word2, str):
        raise ValueError("Both inputs must be strings")

    # Calculate the Levenshtein distance
    distance = edit_distance(word1, word2)

    # Add a small constant offset to ensure the distance is not zero
    distance_with_offset = distance + offset

    return distance_with_offset


def _find_synonym_antonym(phrase: str, nlp_instance):
    """
    Finds the best synonym and antonym for a given phrase based on computed similarity.

    Args:
    phrase (str): The input phrase for which synonyms and antonyms are to be found.
    nlp_instance: spaCy NLP model instance.

    Returns:
    tuple: A tuple containing:
    - best_synonym (str): The most similar synonym to the input phrase.
    - best_antonym (str): The most dissimilar antonym or "not {phrase}" if no antonyms are found.
    """
    logger.info(f"Finding best synonym and antonym for phrase: {phrase}")

    synonyms, antonyms = _find_synonyms_antonyms_with_spacy(phrase, nlp_instance)

    # Finding the Best Synonym
    best_synonym = phrase  # Initialize the best synonym as the original phrase
    if synonyms:
        max_similarity = -1
        for synonym in synonyms:
            if synonym != phrase:  # Avoid comparing the phrase with itself
                similarity = _compute_word_distance(phrase, synonym)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_synonym = synonym

    # Finding the Best Antonym
    best_antonym = f"not {phrase}"  # Initialize the best antonym as "not {phrase}"
    if antonyms:
        min_similarity = float('inf')
        for antonym in antonyms:
            similarity = _compute_word_distance(phrase, antonym)
            if similarity < min_similarity:
                min_similarity = similarity
                best_antonym = antonym

    logger.info(f"Best synonym: {best_synonym}, Best antonym: {best_antonym}")
    return best_synonym, best_antonym


def _replace_word_in_text(text, original_word, new_word):
    """
    Replaces all occurrences of 'original_word' with 'new_word' in the given text.

    Args:
    - text (str): The text in which to replace words.
    - original_word (str): The word to be replaced.
    - new_word (str): The word to replace with.

    Returns:
    - str: The modified text with the word replaced.
    """
    if not isinstance(text, str):
        raise ValueError("The text must be a string")
    if not isinstance(original_word, str) or not isinstance(new_word, str):
        raise ValueError("Both original_word and new_word must be strings")

    logger.debug(f"Replacing '{original_word}' with '{new_word}' in text.")
    # Replace original_word with new_word in the text
    return text.replace(original_word, new_word)


@lru_cache(maxsize=1000)
def _compute_prediction(model, text: str) -> np.ndarray:
    """
    Wrapper for model prediction with caching support.

    Args:
        model: Trained model with `predict_proba` method.
        text (str): Input text.

    Returns:
        np.ndarray: Predicted probabilities for the input text.
    """
    return model.predict_proba([text])


# def compute_text_derivative_matrix(model, X, text_features, data, tau_threshold=0.1, n_jobs=1):
#     """
#     Computes the derivative matrix for text data using a model's predictions.
#
#     Parameters:
#     - model: A trained model with a predict_proba method.
#     - X (scipy.sparse matrix): The feature matrix.
#     - feature_names (list of str): The list of feature names.
#     - data (list of str): The list of text data corresponding to samples in X.
#     - tau_threshold (float): The threshold to decide between first and second-order derivatives.
#     - n_jobs (int): The number of jobs to run in parallel.
#
#     Returns:
#     - scipy.sparse.csr_matrix: The computed derivative matrix.
#     """
#     print("### Start calculating the D matrix for text")
#
#     n_samples, n_features = X.shape
#     J_D = sparse.lil_matrix((n_samples, n_features))  # Initialize as LIL for efficiency
#     H_D = sparse.lil_matrix((n_samples, n_features))  # Initialize as LIL for efficiency
#
#     # Storage for predictions
#     predictions_data = []  # List to store (sample_idx, feature_idx, y_i, y_pos, y_neg)
#
#     def compute_derivative(i, j):
#         original_word = text_features[j]
#         synonym, antonym = find_synonym_antonym(original_word, nlp)
#
#         prediction = model.predict_proba([data[i]])
#
#
#         perturbed_text = replace_word_in_text(data[i], original_word, synonym)
#         prediction_pos = model.predict_proba([perturbed_text])
#
#
#         perturbed_text = replace_word_in_text(data[i], original_word, antonym)
#         prediction_neg = model.predict_proba([perturbed_text])
#
#
#         y_i, y_pos, y_neg = extract_prediction_scalar(prediction, prediction_pos, prediction_neg)
#
#         # Append predictions to the list
#         predictions_data.append((i, j, y_i, y_pos, y_neg))
#
#         J_D[i, j] = (y_pos - y_neg) / compute_word_distance(synonym, antonym)
#         H_D[i, j] = (y_pos - 2 * y_i + y_neg) / compute_word_distance(original_word, synonym) ** 2
#
#     non_zero_indices = X.nonzero()
#
#     Parallel(n_jobs=n_jobs)(delayed(compute_derivative)(i, j) for i, j in zip(*non_zero_indices))
#
#     norm_H_D = np.linalg.norm(H_D.toarray(), 'fro')
#     norm_J_D = np.linalg.norm(J_D.toarray(), 'fro')
#     tau = norm_H_D / (1 + norm_J_D)
#     print(f"Calculated tau: {tau}")
#
#     D = J_D if tau < tau_threshold else H_D
#     print(f"Using {'first-order' if tau < tau_threshold else 'second-order'} derivatives.")
#
#     print("### End calculating the D matrix for text")
#     return D.tocsr()

def _predict_scalar(model, text: str) -> float:
    """
    Predicts the scalar probability for a single text input.

    Args:
        model: A trained model with a `predict_proba` method.
        text (str): Input text sample.

    Returns:
        float: The predicted probability (assumes binary classification).
    """
    prediction = model.predict_proba([text])
    return prediction  # Assuming binary classification


def _compute_single_derivative(
        model, original_word: str, text: str
) -> Tuple[float, float]:
    """
    Computes the first-order and second-order derivatives for a single text feature.

    Args:
        model: A trained model with a `predict_proba` method.
        original_word (str): The word being perturbed.
        text (str): The original text.

    Returns:
        Tuple[float, float]: The first-order and second-order derivatives (J_D, H_D).
    """

    nlp = get_nlp()
    synonym, antonym = _find_synonym_antonym(original_word, nlp)

    # Compute perturbed predictions
    prediction = _predict_scalar(model, text)
    prediction_pos = _predict_scalar(model, _replace_word_in_text(text, original_word, synonym))
    prediction_neg = _predict_scalar(model, _replace_word_in_text(text, original_word, antonym))
    y_i, y_pos, y_neg = extract_prediction_scalar(prediction, prediction_pos, prediction_neg)

    # Compute first-order and second-order derivatives
    j_d = (y_pos - y_neg) / _compute_word_distance(synonym, antonym)
    h_d = (y_pos - 2 * y_i + y_neg) / _compute_word_distance(original_word, synonym) ** 2

    return j_d, h_d


def compute_text_derivative_matrix(
        model, X, text_features, data, tau_threshold=0.1, n_jobs=1
) -> sparse.csr_matrix:
    """
    Computes the derivative matrix for text data using a model's predictions.

    Args:
        model: A trained model with a `predict_proba` method.
        X (scipy.sparse matrix): Feature matrix.
        text_features (List[str]): Feature names (keywords).
        data (List[str]): Input text samples.
        tau_threshold (float): Threshold for deciding first vs. second-order derivatives.
        n_jobs (int): Number of parallel jobs.

    Returns:
        scipy.sparse.csr_matrix: The derivative matrix.
    """
    logger.info("### Start calculating the derivative matrix for text")
    n_samples, n_features = X.shape

    # Initialize sparse matrices for derivatives
    J_D = sparse.lil_matrix((n_samples, n_features))
    H_D = sparse.lil_matrix((n_samples, n_features))

    # Identify non-zero feature indices in the sparse matrix
    non_zero_indices = list(zip(*X.nonzero()))

    # Parallel computation of derivatives
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_single_derivative)(
            model, text_features[j], data[i]
        )
        for i, j in non_zero_indices
    )

    # Populate the sparse matrices with computed results
    for (i, j), (j_d, h_d) in zip(non_zero_indices, results):
        J_D[i, j] = j_d
        H_D[i, j] = h_d

    # Compute norms and decide which derivative matrix to use
    norm_H_D = np.linalg.norm(H_D.toarray(), 'fro')
    norm_J_D = np.linalg.norm(J_D.toarray(), 'fro')
    tau = norm_H_D / (1 + norm_J_D)

    if tau < tau_threshold:
        logger.info(f"Using first-order derivatives (tau={tau})")
        D = J_D
    else:
        logger.info(f"Using second-order derivatives (tau={tau})")
        D = H_D

    logger.info("### End calculating the derivative matrix for text")
    return D.tocsr()
