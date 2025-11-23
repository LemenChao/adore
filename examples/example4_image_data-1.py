# -*- coding: utf-8 -*-
"""
Module Name: example6_image_data-1.py
Description:
    This module demonstrates how to implement ADORE algorithm for black-box model explanation on the CIFAR-10 dataset.
    A VGG16 model is used as the prediction model images for classification. ADORE is then used to explain the model’s
    predictions on a subset of test images. The code also shows how to visualize feature maps to gain insights into
    model behavior.

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
Version: [0.0.1]

License:
    This module is licensed under GPL-3.0 . See LICENSE file for details.
"""
import os
import pickle
import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from sklearn.preprocessing import StandardScaler

from adore import ADORE
from adore.visualizing import visualize_feature_maps

# Set up logger
logger = logging.getLogger(__name__)

# Load CIFAR-10 data
local_data_path = '../data/cifar10_data/'  # Replace with your local path


def load_batch(fpath):
    """ Load a batch of CIFAR-10 dataset """
    with open(fpath, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        X = data_dict[b'data']
        y = data_dict[b'labels']
        X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        y = np.array(y)
        return X, y


def load_cifar10_data(path):
    """ Load entire CIFAR-10 dataset from local directory """
    X_train = []
    y_train = []
    for i in range(1, 6):
        fpath = os.path.join(path, f'data_batch_{i}')
        X, y = load_batch(fpath)
        X_train.append(X)
        y_train.append(y)

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    X_test, y_test = load_batch(os.path.join(path, 'test_batch'))

    return (X_train, y_train), (X_test, y_test)


(X_train, y_train), (X_test, y_test) = load_cifar10_data(local_data_path)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 32 * 32 * 3)).reshape(-1, 32, 32, 3)
X_test_scaled = scaler.transform(X_test.reshape(-1, 32 * 32 * 3)).reshape(-1, 32, 32, 3)

logger.info(f"Training data shape: {X_train_scaled.shape}, Training labels shape: {y_train.shape}")
logger.info(f"Test data shape: {X_test_scaled.shape}, Test labels shape: {y_test.shape}")

# Load pre-trained VGG16 model and adjust for 32x32 images
vgg16 = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Define sequential model and add VGG16 layers
model = Sequential()
for layer in vgg16.layers:
    model.add(layer)

model.add(Flatten())
model.add(Dense(10, activation='softmax', name='predictions'))

print(model.summary())

# Load best weights saved by ModelCheckpoint
# pred_model.load_weights('../examples/weights.keras')


# Select a subset of images for explanation
# np.random.seed(42)
# num_samples = 5
# sample_indices = np.random.choice(X_test_scaled.shape[0], num_samples, replace=False)
# samples_X = X_test_scaled[sample_indices]
# samples_y = y_test[sample_indices]
samples_X = X_test_scaled[0:5]

# Import ADORE and explain sample images
adore=ADORE(model=model, data=samples_X, n_jobs=3)
# Generate explanations for the top 5 features
results=adore.explain(k=5)

# Visualize feature maps
visualize_feature_maps(samples_X, results[6], top_features=5, num_samples=5)