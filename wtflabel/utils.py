from collections import Counter
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image


def load_mnist(size=60000, device=torch.device("cpu")):
    folder = Path("mnist/train")
    img_paths = list(folder.glob("*/*.jpg"))
    img_paths = np.random.choice(img_paths, len(img_paths), replace=False)
    to_tensor = transforms.ToTensor()
    samples_t = []
    labels = []
    for img_path in img_paths:
        img = Image.open(img_path)
        img_t = to_tensor(img)[0, :, :].view(1, -1).to(device)
        samples_t.append(img_t)
        labels.append(int(str(img_path).split('/')[-2]))
        if len(samples_t) == size: break
    return samples_t, labels, np.array([str(p) for p in img_paths])


def entropy(logits):
    probs = F.softmax(logits[0], dim=0).detach().cpu().numpy()
    entropy = -np.sum([i * np.log(i + 1e-10) for i in probs])
    return entropy


def recommend_samples(model, samples, idxs, batchsize=5, max_entropy=True):
    entropies = [entropy(model(samples[i])) for i in idxs]
    sorted_samples = sorted(list(enumerate(entropies)), key=lambda x: -x[1])
    if max_entropy:
        return [idxs[i[0]] for i in sorted_samples[:batchsize]]
    else:
        return [idxs[i[0]] for i in sorted_samples[-batchsize:]]


def recommend_group(model, samples, idxs, batchsize=9):
    logits = [model(samples[i])[0].cpu().detach().numpy() for i in idxs]
    predictions = [np.argmax(l) for l in logits]
    most_freq_label = Counter(predictions).most_common(1)[0][0]
    group = [idxs[i] for i, p in enumerate(predictions) if p == most_freq_label]
    if len(group) > batchsize:
        group = np.random.choice(group, batchsize, replace=False)
    return group, most_freq_label
