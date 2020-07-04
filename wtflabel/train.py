import torch
from torch import nn, optim
import numpy as np


def train(user_model, user_data, dataset, steps=200, batchsize=5, device=torch.device("cpu")):
    print(user_data)
    xs = list(user_data.keys())
    ys = list(user_data.values())
    optimizer = optim.Adam(user_model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    batchsize = min(len(xs), batchsize)
    for t in range(steps):
        optimizer.zero_grad()
        batch_x = []
        batch_y = []
        for s in np.random.choice(len(xs), batchsize, replace=False):
            batch_x.append(dataset[xs[s]])
            batch_y.append(ys[s])
        batch_x = torch.cat(batch_x)
        logits = user_model(batch_x)
        loss = loss_fn(logits, torch.tensor(batch_y).to(device))
        loss.backward()
        optimizer.step()


def validate(user_model, user_data, dataset, batchsize=5, sample_ids=None):
    acc = []
    if sample_ids is None:
        xs = [dataset[i] for i in user_data.keys()]
        ys = list(user_data.values())
    else:
        xs = []
        ys = []
        for i in sample_ids:
            if i in user_data:
                xs.append(dataset[i])
                ys.append(user_data[i])
    for batch in range(np.floor(len(xs)/batchsize)):
        start = batch * batchsize
        end = start + batchsize
        logits = user_model(torch.cat(xs[start:end]))
        pred = np.argmax(logits.detach().cpu().numpy(), axis=1)
        acc.append(np.mean(pred == ys[start:end]))
    return np.mean(acc)
