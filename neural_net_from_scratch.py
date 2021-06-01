from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)
print(y_train.shape)

import torch

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())

import math

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x):
    #x.shape is [64,10]
    return x - x.exp().sum(-1).log().unsqueeze(-1)
    #return的也为[64,10]

def model(xb):
    #xb.shape is [64,784]
    #weights.shape is [784,10]
    #bias.shape is [10]
    return log_softmax(xb @ weights + bias)

bs = 64  # batch size

xb = x_train[0:bs]  # a mini-batch from x
#xb.shape is [64,784]
preds = model(xb)  # predictions
print(preds[0], preds.shape)
#tensor([-2.7085, -1.8097, -2.5516, -2.7430, -2.4519, -2.1625, -2.4260, -2.5655,
#        -2.5773, -1.6879], grad_fn=<SelectBackward>) torch.Size([64, 10])

def nll(input, target):
    return -input[range(target.shape[0]), target].mean()
#让preds在正确的索引处值越大越好

loss_func = nll

yb = y_train[0:bs]
print(loss_func(preds, yb))
#preds.shape is [64,10]
#yb.shape is [64]

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)#选出横向值最大的索引
    return (preds == yb).float().mean()

print(accuracy(preds, yb))

from IPython.core.debugger import set_trace

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

print(loss_func(model(xb), yb), accuracy(model(xb), yb))
