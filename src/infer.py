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

from embed import series_to_tensor, preprocess
from cnn import CNNText
import changelog
import numpy as np
import pandas as pd
import itertools
import os
import torch
from tqdm import trange, tqdm
import sklearn.metrics as metrics
import sys
from pandarallel import pandarallel

# Initialization
pandarallel.initialize()

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
#  causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

# torch.set_num_threads(16)

# Basic Parameters
THRESHOLD = 0.5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

batch_size = 512  # how many samples to process at once


changelog.write_csv(sys.argv[1:])
data = pd.read_csv("data/data.csv")
# limit the input size when testing
# data = data.iloc[range(100)]

data["text"] = data["text"].parallel_apply(preprocess)
commits = series_to_tensor(data["text"])
valid_loader = torch.utils.data.DataLoader(
    commits, batch_size=batch_size, shuffle=False
)

assert os.path.exists("data/textcnn_model")
model = torch.load("data/textcnn_model").to(device)
mlb = model.mlb
n_labels = len(model.mlb.classes_)


def tensor_to_class(val_preds):
    """converts a tensor of 0/1 to class labels. 1 indicates a prediction"""
    outputs = torch.sigmoid(val_preds)
    outputs = outputs.cpu().detach().numpy()
    results = []
    y_pred = np.where(outputs > THRESHOLD, 1, 0)
    # a 0/1 np array with shape number of samples * number of classes
    for i in range(y_pred.shape[0]):
        results.append(list(itertools.compress(mlb.classes_, y_pred[i])))
    return results


# a list of lists storing the class predictions for each input commit.
class_preds = []
with torch.no_grad():
    for i, (x_batch) in enumerate(tqdm(valid_loader)):
        # The operation is not in-place, and so the reassignment is required.
        x_batch = x_batch.to(device)
        y_pred = model.forward(x_batch).detach()
        class_preds += tensor_to_class(y_pred)
        # val_preds[i * batch_size : (i + 1) * batch_size] = y_pred


for path in sys.argv[1:]:
    changelog.write_changelog(path, class_preds)
