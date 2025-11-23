# -*- coding: utf-8 -*-
"""
Module Name: preprocessing.py
Description:
    This module provides a set of general-purpose data preprocessing functions. It is designed to handle a wide variety of input data types, including text, tabular, and image data.
    It contains functions for handling missing values, encoding categorical data, scaling numerical features, normalizing text, and converting image formats.

Author:
    Fang Anran <fanganran97@126.com>
    Lei Ming <leimingnick@ruc.edu.cn>


Maintainer:
    Fang Anran <fanganran97@126.com>
    Lei Ming <leimingnick@ruc.edu.cn>


Contributors:
    Chao Lemen <chaolemen@ruc.edu.cn>
    Lei Ming <leimingnick@ruc.edu.cn>
    Fang Anran <fanganran97@126.com>

Created on: 2025-1-2
Last Modified on: 2025-1-9
Version: [0.1.2]

License:
    This module is licensed under GPL-3.0 . See LICENSE file for details.

Usage Example:
    # >>> from adore.preprocessing import preprocess_table_data, preprocess_text_data, preprocess_image_data
"""
import logging
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from spellchecker import SpellChecker
from tensorflow.keras.preprocessing import image as keras_image

import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import tensorflow_datasets as tfds
import os

# Get the logger for this module
logger = logging.getLogger(__name__)


# 1. tabular data preprocessing
def _convert_to_dataframe(data):
    """
    Converts the input data to a pandas DataFrame if it is not already in that format.

    Args:
    data (various): The input data to be converted. It can be a list of lists, a dictionary,
                    a NumPy array, or an existing pandas DataFrame.

    Returns:
    pd.DataFrame: The data converted to a pandas DataFrame.
    """
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, dict):
        return pd.DataFrame(data)
    elif isinstance(data, list):
        return pd.DataFrame(data)
    elif hasattr(data, 'shape'):  # Handle numpy arrays or other array-like structures
        return pd.DataFrame(data)
    else:
        logger.error(
            "Unsupported data format. Please provide data in a format that can be converted to a DataFrame.")


def _fill_missing_values(df, strategy='mean', columns=None):
    """
    Fills missing values in specified columns of the DataFrame using the given strategy.
    The strategy can be 'mean', 'median', or 'most_frequent'.
    By default, it fills missing values with the mean of the column.

    Args:
    df (pd.DataFrame): The input DataFrame containing data.
    strategy (str, optional): The strategy for filling missing values. Options are 'mean', 'median', 'most_frequent'. Default is 'mean'.
    columns (list of str, optional): The list of column names where missing values should be filled. If None, it applies to all columns.

    Returns:
    pd.DataFrame: The DataFrame with missing values filled in specified columns.
    """
    imputer = SimpleImputer(strategy=strategy)
    df[columns] = imputer.fit_transform(df[columns])
    return df


def _encode_categorical_data(df, columns):
    """
    Encodes categorical columns into numerical values using label encoding.
    Each unique category in a column is assigned a corresponding integer.

    Args:
    df (pd.DataFrame): The input DataFrame containing categorical data.
    columns (list of str): The list of column names that contain categorical data to be encoded.

    Returns:
    pd.DataFrame: The DataFrame with categorical columns encoded as integers.
    """
    encoder = LabelEncoder()
    for col in columns:
        df[col] = encoder.fit_transform(df[col])
    return df


def _normalize_data(df, columns):
    """
    Normalizes the numerical columns by scaling them to have a mean of 0 and a standard deviation of 1.
    This is done using the StandardScaler from sklearn.

    Args:
    df (pd.DataFrame): The input DataFrame containing numerical data.
    columns (list of str): The list of column names to normalize.

    Returns:
    pd.DataFrame: The DataFrame with normalized numerical columns.
    """
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def preprocess_table_data(data, missing_values_columns, categorical_columns, numerical_columns):
    """
    Preprocesses the table data by handling missing values, encoding categorical columns,
    and normalizing numerical columns.

    It performs the following operations:
    - Fills missing values in specified columns.
    - Encodes categorical columns into numerical values.
    - Normalizes numerical columns to have zero mean and unit variance.

    Args:
    df (pd.DataFrame): The input DataFrame containing the data.
    missing_values_columns (list of str): The list of columns with missing values to be filled.
    categorical_columns (list of str): The list of categorical columns to be encoded.
    numerical_columns (list of str): The list of numerical columns to be normalized.

    Returns:
    pd.DataFrame: The preprocessed DataFrame after applying all transformations.
    """
    df = _convert_to_dataframe(data)
    df = _fill_missing_values(df, strategy='mean', columns=missing_values_columns)
    df = _encode_categorical_data(df, columns=categorical_columns)
    df = _normalize_data(df, columns=numerical_columns)
    return df


# 2. text preprocessing
# Initialize lemmatizer and spell checker
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()


def _lowercase_text(text):
    """
    Converts the input text to lowercase to avoid case sensitivity in further processing.

    Args:
    text (str): The input text to be converted.

    Returns:
    str: The lowercase version of the input text.
    """
    return text.lower()


def _remove_punctuation(text):
    """
    Removes punctuation and special characters from the input text.
    Keeps only alphanumeric characters and spaces.

    Args:
    text (str): The input text to be cleaned.

    Returns:
    str: The text with punctuation removed.
    """
    return re.sub(r'[^\w\s]', '', text)


def _remove_extra_whitespace(text):
    """
    Removes extra whitespace from the input text by collapsing multiple spaces
    into a single space and stripping leading/trailing spaces.

    Args:
    text (str): The input text with extra whitespace.

    Returns:
    str: The text with extra whitespace removed.
    """
    return re.sub(r'\s+', ' ', text).strip()


def _remove_stop_words(text):
    """
    Removes stop words (common words like 'the', 'and', etc.) from the text
    to focus on more meaningful words for keyword extraction.

    Args:
    text (str): The input text to remove stop words from.

    Returns:
    str: The text with stop words removed.
    """
    return ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])


def _lemmatize_text(text):
    """
    Lemmatizes the words in the input text using WordNetLemmatizer.
    Lemmatization reduces words to their base or root form, preserving their meaning.

    Args:
    text (str): The input text to be lemmatized.

    Returns:
    str: The lemmatized text.
    """
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])


def _remove_numbers(text):
    """
    Removes all numeric characters from the input text.

    Args:
    text (str): The input text to remove numbers from.

    Returns:
    str: The text with numbers removed.
    """
    return re.sub(r'\d+', '', text)


def _correct_spelling(text):
    """
    Corrects the spelling of words in the text using the PySpellChecker library.

    Args:
    text (str): The input text with potential spelling errors.

    Returns:
    str: The text with corrected spelling.
    """
    words = text.split()
    corrected_words = [spell.correction(word) for word in words]
    return ' '.join(corrected_words)


def _remove_html_tags(text):
    """
    Removes HTML tags from the input text if any, using a regular expression to match and remove HTML tags.

    Args:
    text (str): The input text possibly containing HTML tags.

    Returns:
    str: The text with HTML tags removed.
    """
    return re.sub(r'<.*?>', '', text)


def _filter_text(text, banned_words):
    """
    Removes any banned or unwanted words from the input text.

    Args:
    text (str): The input text to be filtered.
    banned_words (list of str): A list of words to be filtered out.

    Returns:
    str: The filtered text with banned words removed.
    """
    return ' '.join([word for word in text.split() if word not in banned_words])


def _preprocess_single_text(text):
    """
    A helper function that applies the preprocessing steps to a single text.

    Args:
    text (str): The input text to be preprocessed.

    Returns:
    str: The cleaned and preprocessed text.
    """
    text = _lowercase_text(text)
    text = _remove_punctuation(text)
    text = _remove_extra_whitespace(text)
    text = _remove_stop_words(text)
    text = _lemmatize_text(text)
    text = _remove_numbers(text)
    text = _remove_html_tags(text)
    text = _correct_spelling(text)
    return text


def preprocess_text_data(texts):
    """
    A comprehensive preprocessing function that applies multiple cleaning steps to a list of input texts:
    1. Converts text to lowercase
    2. Removes punctuation and special characters
    3. Removes extra whitespaces
    4. Removes stop words
    5. Lemmatizes the text
    6. Removes numbers
    7. Corrects spelling
    8. Removes HTML tags

    Args:
    texts (list of str): A list of input texts to be preprocessed.

    Returns:
    list of str: A list of cleaned and preprocessed texts ready for keyword extraction.
    """
    # Apply preprocessing to each text in the list
    preprocessed_texts = [_preprocess_single_text(text) for text in texts]
    return preprocessed_texts


def preprocess_text_data(texts):
    """
    A comprehensive preprocessing function that applies multiple cleaning steps to a list of input texts:
    1. Converts text to lowercase
    2. Removes punctuation and special characters
    3. Removes extra whitespaces
    4. Removes stop words
    5. Lemmatizes the text
    6. Removes numbers
    7. Corrects spelling
    8. Removes HTML tags

    Args:
    texts (list of str): A list of input texts to be preprocessed.

    Returns:
    list of str: A list of cleaned and preprocessed texts ready for keyword extraction.
    """
    # Apply preprocessing to each text in the list
    preprocessed_texts = [_preprocess_single_text(text) for text in texts]
    return preprocessed_texts


# 3. image preprocessing
# Global Variables
DEFAULT_IMAGE_SIZE = (224, 224)  # Adjustable as needed
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def resize_image(image, target_size=DEFAULT_IMAGE_SIZE):
    """Resize the image to the target size."""
    return tf.image.resize(image, target_size)


def convert_channels(image, num_channels=3):
    """
    Convert the image to the desired number of channels.
    - If grayscale, convert to the specified number of channels.
    - If channels exceed the desired number, remove the extra channels.
    """
    current_channels = tf.shape(image)[-1]
    if current_channels == 1 and num_channels == 3:
        image = tf.image.grayscale_to_rgb(image)
    elif current_channels > num_channels:
        image = image[..., :num_channels]
    elif current_channels < num_channels:
        # If more channels are needed, replicate existing channels (e.g., grayscale to RGB)
        image = tf.image.grayscale_to_rgb(image)
    return image


def normalize_image(image, normalization='0-1'):
    """
    Normalize the image.

    Args:
        image: Tensor, image data.
        normalization: str, normalization method, options: '0-1', '0-255', '-1-1'.

    Returns:
        Tensor, normalized image.
    """
    if normalization == '0-1':
        image = image / 255.0
    elif normalization == '0-255':
        image = tf.cast(image, tf.float32)
    elif normalization == '-1-1':
        image = (image / 127.5) - 1.0
    else:
        raise ValueError("Unsupported normalization. Choose from '0-1', '0-255', '-1-1'.")
    return image


def preprocess_image(image, target_size=DEFAULT_IMAGE_SIZE, num_channels=3, normalization='0-1'):
    """
    Generic image preprocessing function including resizing, channel conversion, and normalization.

    Args:
        image: Tensor, image data.
        target_size: tuple, desired image size.
        num_channels: int, desired number of channels.
        normalization: str, normalization method.

    Returns:
        Tensor, preprocessed image.
    """
    image = resize_image(image, target_size)
    image = convert_channels(image, num_channels)
    image = normalize_image(image, normalization)
    return image


def load_and_preprocess_single_image(image_path, target_size=DEFAULT_IMAGE_SIZE, num_channels=3, normalization='0-1'):
    """
    Load and preprocess a single local image.

    Args:
        image_path: str, path to the image file.
        target_size: tuple, desired image size.
        num_channels: int, desired number of channels.
        normalization: str, normalization method.

    Returns:
        NumPy array, preprocessed image with shape (1, height, width, channels).
    """
    img = keras_image.load_img(image_path)
    img_array = keras_image.img_to_array(img)
    img_tensor = tf.convert_to_tensor(img_array)
    preprocessed_img = preprocess_image(image=img_tensor, target_size=target_size, num_channels=num_channels,
                                        normalization=normalization)
    preprocessed_img = tf.expand_dims(preprocessed_img, axis=0)  # Add batch dimension
    return preprocessed_img.numpy()


def preprocess_dataset(dataset, target_size=DEFAULT_IMAGE_SIZE, num_channels=3, normalization='0-1', shuffle=False):
    """
    Generic dataset preprocessing function.

    Args:
        dataset: tf.data.Dataset, containing images and labels.
        target_size: tuple, desired image size.
        num_channels: int, desired number of channels.
        normalization: str, normalization method.
        shuffle: bool, whether to shuffle the data.

    Returns:
        tf.data.Dataset, preprocessed dataset.
    """

    def _preprocess(image, label):
        image = preprocess_image(image, target_size, num_channels, normalization)
        return image, label

    if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.map(_preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def load_mnist_dataset(target_size=DEFAULT_IMAGE_SIZE, num_channels=3, normalization='0-1', shuffle=True):
    """
    Load and preprocess the MNIST dataset.

    Args:
        target_size: tuple, desired image size.
        num_channels: int, desired number of channels.
        normalization: str, normalization method.
        shuffle: bool, whether to shuffle the data.

    Returns:
        train_ds, test_ds: tf.data.Dataset, preprocessed training and testing datasets.
    """
    (train_ds, test_ds), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=shuffle,
        as_supervised=True,
        with_info=True
    )

    def _preprocess_mnist(image, label):
        # MNIST is grayscale with 1 channel
        return preprocess_image(image, target_size, num_channels, normalization), label

    if shuffle:
        train_ds = train_ds.shuffle(buffer_size=10000)

    train_ds = train_ds.map(_preprocess_mnist, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(_preprocess_mnist, num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return train_ds, test_ds


def load_cifar10_dataset(target_size=DEFAULT_IMAGE_SIZE, num_channels=3, normalization='0-1', shuffle=True):
    """
    Load and preprocess the CIFAR-10 dataset.

    Args:
        target_size: tuple, desired image size.
        num_channels: int, desired number of channels.
        normalization: str, normalization method.
        shuffle: bool, whether to shuffle the data.

    Returns:
        train_ds, test_ds: tf.data.Dataset, preprocessed training and testing datasets.
    """
    (train_ds, test_ds), ds_info = tfds.load(
        'cifar10',
        split=['train', 'test'],
        shuffle_files=shuffle,
        as_supervised=True,
        with_info=True
    )

    train_ds = preprocess_dataset(train_ds, target_size, num_channels, normalization, shuffle)
    test_ds = preprocess_dataset(test_ds, target_size, num_channels, normalization, shuffle=False)

    return train_ds, test_ds