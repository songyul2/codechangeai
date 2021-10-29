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

from embed import preprocess, series_to_tensor
import sys
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import time
import pandas as pd
import multiprocessing
import torch

import os


# cell 2
# Basic Parameters
debug = 1
device = torch.device("cpu")
multilabel = 1
n_splits = 5  # Number of K-fold Splits

start_time = time.time()

data = pd.read_csv("data/data.train.csv")
data.info(verbose=True)
# remove NULL Values from data
data.dropna(inplace=True)
# data = data.loc[data.label != "stable guest abi"]

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

data["text"] = data["text"].apply(preprocess)

train_X, test_X, train_y, test_y = train_test_split(
    data["text"], data["label"], stratify=None, test_size=1 / n_splits
)

# cell 25
print(f"Train shape : { train_X.shape}")
print(f"Test shape : { test_X.shape}")

# One-hot encode data
mlb = MultiLabelBinarizer()
train_y = mlb.fit_transform(train_y.values)
test_y = mlb.transform(test_y.values)
print("label classes", mlb.classes_)

# model = gensim.models.FastText(
#    sentences=train_X.tolist(), workers=multiprocessing.cpu_count(), vector_size=embed_size)

x_train = series_to_tensor(train_X, train=1)
x_cv = series_to_tensor(test_X)
y_train = torch.tensor(train_y, dtype=torch.float)
y_cv = torch.tensor(test_y, dtype=torch.float)
start_time = time.time()
# Create Torch datasets
train = torch.utils.data.TensorDataset(x_train, y_train)
valid = torch.utils.data.TensorDataset(x_cv, y_cv)
d = {"train": train, "valid": valid, "mlb": mlb}
torch.save(d, "data/datasets.pt")
