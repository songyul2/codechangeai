# IMPORTS
# import dataset
from cnn import CNNText
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # using a backend that doesn't display to the user, e.g. 'Agg'
import os
import torch
from tqdm import trange, tqdm
import sklearn.metrics as metrics
from sklearn.metrics import classification_report

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
#  causes cuDNN to benchmark multiple convolution algorithms and select the fastest.


# cell 2
# Basic Parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
resume_training = 0
n_epochs = 10  # how many times to iterate over all samples
batch_size = 512  # how many samples to process at once

n_splits = 5  # Number of K-fold Splits
lr = 5e-3  # learning rate
debug = 0

try:
    datasets = torch.load("data/datasets.pt")
except FileNotFoundError as fnf_error:
    print(fnf_error)

# dict preserves order
train, valid, le = datasets.values()
n_labels = len(le.classes_)
# Each sample will be retrieved by indexing tensors along the first
# dimension.
__, y_cv = valid[:]
test_y = y_cv.cpu().numpy()
# Create Data Loaders
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

train_loss = []
valid_loss = []


def valid_model(model, valid_loader):
    # Set model to validation configuration -Doesn't get trained here
    model.eval()
    avg_val_loss = 0.0
    val_preds = torch.zeros((len(valid_loader.dataset), n_labels))
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            # The operation is not in-place, and so the reassignment is required.
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model.forward(x_batch).detach()
            avg_val_loss += model.loss_fn(y_pred, y_batch).item() / len(valid_loader)
            # keep/store predictions
            val_preds[i * batch_size : (i + 1) * batch_size] = y_pred
    return val_preds, avg_val_loss


if resume_training:
    assert os.path.exists("data/textcnn_model")
    model = torch.load("data/textcnn_model").to(device)
else:
    model = CNNText(n_labels).to(device)

start_time = time.time()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=lr
)

for epoch in trange(n_epochs):
    # Set model to train configuration
    model.train()
    avg_loss = 0.0
    train_correct = 0
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        # Predict/Forward Pass
        y_pred = model(x_batch)
        # Compute loss
        loss = model.loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)

        torch.cuda.empty_cache()

    train_loss.append(avg_loss)
    val_preds, avg_val_loss = valid_model(model, valid_loader)
    valid_loss.append(avg_val_loss)
    min_valid_loss_epoch = np.array(valid_loss).argmin()
    # early stopping
    if epoch - min_valid_loss_epoch > 5:
        break

    if epoch % 5 == 4:
        elapsed_time = time.time() - start_time
        print(
            "Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f}  \t time={:.4f}s".format(
                epoch + 1,
                n_epochs,
                avg_loss,
                avg_val_loss,
                elapsed_time,
            )
        )
        print("-" * 55)
        start_time = time.time()

# cell 37
torch.save(model, "data/textcnn_model")


def plot_graph(epochs):
    plt.figure(figsize=(12, 12))
    plt.title("Train/Validation Loss")
    plt.plot(list(np.arange(epochs) + 1), train_loss, label="train")
    plt.plot(list(np.arange(epochs) + 1), valid_loss, label="validation")
    plt.xlabel("num_epochs", fontsize=12)
    plt.ylabel("loss", fontsize=12)
    plt.legend(loc="best")
    plt.savefig("data/Train Validation Loss")


# cell 39
plot_graph(len(train_loss))


outputs = torch.sigmoid(val_preds)
outputs = outputs.cpu().detach().numpy()
roc_metrics = []
THRESHOLD = 0.5
y_pred = np.where(outputs > THRESHOLD, 1, 0)
with open("data/classification_report.txt", "w") as f:
    f.write(
        classification_report(test_y, y_pred, target_names=le.classes_, zero_division=0)
    )

for i in range(n_labels):
    if test_y[:, i].sum() > 1:
        # Stands for One-vs-one. Computes the average AUC of all possible pairwise combinations of classes [5]. Insensitive to class imbalance when average == 'macro'.
        roc = metrics.roc_auc_score(test_y[:, i], outputs[:, i], multi_class="ovo")
        roc_metrics.append(roc)
    else:
        print(le.classes_[i], test_y[:, i].sum())
        roc_metrics.append(0)  # not applicable

s = pd.Series(roc_metrics, index=le.classes_)
fig = plt.figure()
s.plot(
    kind="bar",
    figsize=(20, 11.25),
    title="roc auc score per class on test data",
    rot=90,
    grid=True,
    fontsize=15,
)
plt.savefig("data/roc auc score", bbox_inches="tight")
# Bounding box in inches: only the given portion of the figure is saved. If 'tight', try to figure out the tight bbox of the figure.
