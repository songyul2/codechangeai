# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See LICENSE for more details.
#
# Copyright: Red Hat Inc. 2021
# Author: Songyu Liu <sonliu@redhat.com>

from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.base import BaseEstimator
from sklearn import utils as skl_utils
from tqdm import tqdm
from gensim.parsing.preprocessing import preprocess_string
import multiprocessing
import numpy as np
import sys
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk import word_tokenize
import time
import pandas as pd
import re

from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn import svm
import multiprocessing

import numpy as np
import os
import nltk

# nltk.download("punkt")  # run this only once


# cell 2
# Basic Parameters
multilabel = 1
n_splits = 5  # Number of K-fold Splits
embed_size = 300
start_time = time.time()

data = pd.read_csv("data/data.csv")
data.info(verbose=True)
# remove NULL Values from data
data.dropna(inplace=True)
data = data.loc[data.label != "stable guest abi"]

if multilabel:

    def combine(group):
        """keep the 0th text and combine the labels"""
        # each label has to be an iterable
        combined = pd.DataFrame(
            {"text": group.iloc[0, 0], "label": [set(group["label"])]}
        )
        if group.iloc[0, 0].startswith("update commit from"):
            print(combined)
            sys.exit("this message should not be considered")
        return combined

    grouped = data.groupby("text")
    data = grouped.apply(combine)
    # apply will then take care of combining the results back together into a single dataframe or series.

else:
    data = data.drop_duplicates(["text"])
    data["label"] = data["label"].apply(lambda x: [x])

data["text"] = data["text"].apply(preprocess_string)  # Return type    list of str

data.to_pickle("data/preprocessed")
print(start_time - time.time())


class Doc2VecTransformer:
    def __init__(self, vector_size=embed_size, learning_rate=0.02, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._model = None
        self.vector_size = vector_size
        self.workers = multiprocessing.cpu_count()
        # Multiple cores not being used in Word2Vec

    def fit(self, df_x, df_y=None):
        tagged_x = [TaggedDocument((row), [index]) for index, row in enumerate(df_x)]
        model = Doc2Vec(
            documents=tagged_x, vector_size=self.vector_size, workers=self.workers
        )

        for epoch in tqdm(range(self.epochs)):
            model.train(
                skl_utils.shuffle(tagged_x),
                total_examples=len(tagged_x),
                epochs=self.epochs,
            )

            model.alpha = self.learning_rate
            model.min_alpha = model.alpha
            # alpha (float, optional) – The initial learning rate.
            # min_alpha (float, optional) – Learning rate will linearly drop to min_alpha as training progresses.

        self._model = model
        return self

    def transform(self, df_x):
        return np.asmatrix(
            ([self._model.infer_vector((row)) for index, row in enumerate(df_x)])
        )


train_X_series, test_X, train_y, test_y = train_test_split(
    data["text"], data["label"], stratify=None, test_size=1 / n_splits
)

print(f"Train shape : { train_X_series.shape}")
print(f"Test shape : { test_X.shape}")
transformer = Doc2VecTransformer().fit(train_X_series)
train_X = transformer.transform(train_X_series)
test_X = transformer.transform(test_X)

"""
np.savez('data/doc2vec.npz', train_X, test_X)
arrays = np.load('data/doc2vec.npz')
train_X, test_X = [arrays[name] for name in arrays.files]
"""
scaler = StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)
print(test_X.mean(axis=0))
breakpoint()

# One-hot encode data
mlb = MultiLabelBinarizer()
train_y = mlb.fit_transform(train_y.values)
test_y = mlb.transform(test_y.values)
print("label classes", mlb.classes_)

SVM = svm.LinearSVC(C=1)
# n_jobs=-1 means using all available processes / threads.
clf = MultiOutputClassifier(SVM, n_jobs=-1).fit(train_X, train_y)
y_pred = clf.predict(train_X)
report = classification_report(
    train_y, y_pred, target_names=mlb.classes_, zero_division=0
)
print(report)
y_pred = clf.predict(test_X)
report = classification_report(
    test_y, y_pred, target_names=mlb.classes_, zero_division=0
)
print(report)
with open("data/classification_report_svm.txt", "w") as f:
    f.write(report)
