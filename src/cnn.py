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

import torch.nn as nn
import torch.nn.functional as F
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
#  causes cuDNN to benchmark multiple convolution algorithms
#  and DataLoader the fastest.

# Pytorch Model - TextCNN


class CNNText(nn.Module):
    def __init__(self, mlb):
        super().__init__()
        self.mlb = (
            mlb  # mlb is the MultiLabelBinarizer that has been fitted on the labels
        )
        n_labels = len(mlb.classes_)
        filter_sizes = [2, 3, 4, 5]
        #filter_sizes = [2]
        num_filters = 50
        d = torch.load("data/word_to_tensor.pt")
        word_to_index, embedding_matrix = d.values()
        max_features, embed_size = embedding_matrix.shape
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32), requires_grad=True
        )

        # This loss combines a Sigmoid layer and the BCELoss in one single class.
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (K, embed_size)) for K in filter_sizes]
        )
        # Conv2d(in_channels, out_channels, kernel_size, 
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, n_labels)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  
        # add a dim at index 1. batch size * 1 * commit length * embed_size
        # after conv the last dim is always 1. it is then removed
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        # maxpool along dim 2. dim 2 is then removed
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        # batch size * out_channels

        x = torch.cat(x, 1)  # concatenate along index 1

        x = self.dropout(x)
        return self.fc1(x)
        
