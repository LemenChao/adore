# -*- coding: utf-8 -*-
"""
Module Name: example3_text_data-1.py
Description:
    This module simulates the entire process of training, predicting, and interpreting a machine learning model on textual data using the IMDB dataset as an example, demonstrating the effectiveness of the ADORE algorithm for in-depth analysis and evaluation of the model's performance. Finally, feature and sample-level contributions are revealed, and visualizations help to more intuitively understand the decision logic and important drivers of the model.

    Key functionalities include:
        - **Model Training**: training on sentiment classification tasks with the BERT model using a small training set with balanced sampling. A custom TextDataset class is used to preprocess the text data and the Trainer API is used to load the local BERT pre-trained model for supervised fine-tuning. After the training is completed, the model and the classifier are saved for subsequent prediction.
        - **Model Prediction**: Load the trained BERT model and the classifier through the encapsulated TextClassifier class to predict the sentiment of the input text. The prediction method supports batch input and returns the probability distribution of each category, which is used to achieve positive and negative sentiment classification of movie reviews.        - **Model Explanation with ADORE:** Leveraging the ADORE framework to explain predictions from multiple models, providing detailed feature contributions for samples.
        - **Model Interpretation using ADORE**: predictions are interpreted using the ADORE framework to provide detailed feature contributions to the sample.
        - **Visualization**: primarily plots the feature importance of different samples.
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
# --------------------------------
import json
import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from adore import ADORE


# Set up logger
logger = logging.getLogger(__name__)


# #---------------------------------------Model Prediction--------------------------------------------------------


class TextClassifier:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            logits = self.model(**encodings).logits
        return logits.cpu().numpy()

    def predict_proba(self, texts):
        logits = self.predict(texts)
        return torch.softmax(torch.tensor(logits), dim=1).numpy()


# #---------------------------------------Explanation Example--------------------------------------------------------


model_dir = "../data/models/imdb-finetuned-bert-base-uncased"
classifier = TextClassifier(model_dir)

test_texts = ["Smart dialogue, stunning visuals, and a memorable soundtrack — one of the best films I’ve seen this year."]
print(classifier.predict_proba(test_texts))

adore = ADORE(model=classifier, data=test_texts, n_jobs=1)
results = adore.explain()
feature_names = adore.extract_text_features()
print(feature_names)

# adore_contributions_df = pd.DataFrame(results[0], columns=feature_names)
# adore_contributions_df.to_csv('./results/output/adore_contributions(text)all.csv', index=False)

contributions = results[0]  # Feature contribution matrix
derivative_matrix = results[1]
EACM = results[2]
ERCM = results[3]
SACM = results[4]
SRCM = results[5]
FACM = results[6]
FRCM = results[7]

# Output explanation results
logger.info("Feature contributions for each sample (Simple Model):")
print(contributions)
logger.info("Absolute and relative contribution of the samples")
print(SACM, SRCM, sep='\n')
logger.info("Absolute and relative contribution of the features")
print(FACM, FRCM, sep='\n')

# #---------------------------------------Visualization--------------------------------------------------------

from adore.visualizing import plot_text_contributions
plot_text_contributions(FACM, feature_names, label='Bert', top_k=5)

# for i in range(len(contributions)):
#     plot_text_contributions(contributions[i], feature_names, label='Bert', top_k=5)
