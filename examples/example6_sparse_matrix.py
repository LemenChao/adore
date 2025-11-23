# -*- coding: utf-8 -*-
"""
Module Name: example6_sparse_matrix.py
Description:
    This module uses ADORE to generate explanations for logistic regression and SVM models, respectively, for a given sample document and outputs the respective contribution values to verify the reasonableness and validity of ADORE's explanations on sparse matrix features.

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
Version: [0.0.1]

License:
    This module is licensed under GPL-3.0 . See LICENSE file for details.
"""

import logging
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from adore import ADORE

# Set up logger
logger = logging.getLogger(__name__)

# Define model and vectorizer file names
model_filename_lr = 'logistic_regression_model.joblib'
model_filename_svm = 'svm_model.joblib'
vectorizer_filename = 'tfidf_vectorizer.joblib'

# Sample documents
document_1 = [
    "From: john@example.com Subject: Urgent: Project Deadline Approaching. We need to finalize the report by next Monday. The project is facing some critical issues and requires immediate attention.",
    "From: mary@example.com Subject: Important Update: New Meeting Scheduled for Tomorrow at 3 PM. We will discuss the recent progress on the Q3 financial report and address any remaining concerns."
]

# Load or create TF-IDF vectorizer
if os.path.exists(vectorizer_filename):
    print("Loading the saved vectorizer...")
    vectorizer = joblib.load(vectorizer_filename)
else:
    print("Creating and saving a new vectorizer...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    vectorizer.fit(document_1)  # Fit on the sample documents
    joblib.dump(vectorizer, vectorizer_filename)

X_doc = vectorizer.transform(document_1)  # Transform document_1 into feature vectors

# Output the vocabulary after vectorization to ensure feature matching
# Get feature names
feature_names = vectorizer.get_feature_names_out()
print(f"Feature vocabulary: {feature_names }")

# Load or train models
if os.path.exists(model_filename_lr) and os.path.exists(model_filename_svm):
    logger.info("Loading the saved models...")
    model_lr = joblib.load(model_filename_lr)  # Load the logistic regression model
    model_svm = joblib.load(model_filename_svm)  # Load the SVM model
else:
    logger.info("Training models for the first time and saving them...")
    # Create logistic regression and SVM models
    model_lr = LogisticRegression(max_iter=1000)
    model_svm = SVC(kernel='linear', probability=True)

    # Train the models
    model_lr.fit(X_doc, [0, 1])  # Assume class labels are 0 and 1
    model_svm.fit(X_doc, [0, 1])

    # Save the trained models
    joblib.dump(model_lr, model_filename_lr)
    joblib.dump(model_svm, model_filename_svm)

# Use ADORE to explain the contributions for both documents
adore_lr = ADORE(model=model_lr, data=X_doc)
adore_svm = ADORE(model=model_svm, data=X_doc)

contributions_lr = adore_lr.explain()
contributions_svm = adore_svm.explain()

# Output the contribution values to check their reasonableness
logger.info("Contributions from Logistic Regression:")
print(contributions_lr)
logger.info("Contributions from SVM:")
print(contributions_svm)
