from typing import Optional, Dict, List
import random
from collections import OrderedDict, Counter
import torch
from fastapi import FastAPI, status
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

import wtflabel


app = FastAPI()


origins = [
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


state = hex(random.getrandbits(128))[2:7]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mnist, real_labels, img_paths = wtflabel.utils.load_mnist(5000, device)
models = {}
datasets = {}
accuracies = {}


class Labels(BaseModel):
    state_id: str
    user_id: str
    labels: Dict[str, int]


@app.get("/")
def get_root():
    return {
        "title": "Wow That's Fast Labeling",
        "stateId": state,
    }


@app.get("/samples")
def get_samples(state_id:str, user_id: str, batchsize: Optional[int]=5):
    if state_id != state:
        return "Wrong state ID"
    candidates = np.random.choice(len(mnist), 100, replace=False)
    samples = wtflabel.utils.recommend_samples(
        model=models[user_id],
        samples=mnist,
        idxs=candidates,
        batchsize=batchsize,
    )
    return {"samples": list(img_paths[samples]), "sampleIds": samples}


@app.get("/samplesGroup")
def get_samples_group(state:str, user_id: str, batchsize: Optional[int]=9):
    if state_id != state:
        return "Wrong state ID"
    candidates = np.random.choice(len(mnist), 100, replace=False)
    samples, pred_label = wtflabel.utils.recommend_group(
        model=models[user_id],
        samples=mnist,
        idxs=candidates,
        batchsize=batchsize,
    )
    return {"samples": list(img_paths[samples]), "sampleIds": samples, "predLabel": pred_label}


@app.get("/init")
def init():
    user_id = hex(random.getrandbits(128))[2:7]
    models[user_id] = wtflabel.models.BasicNet().to(device)
    datasets[user_id] = OrderedDict()
    accuracies[user_id] = 0.1
    return {"userId": user_id, "stateId": state}


@app.get("/checkUser")
def check_user(state:str, user_id: str):
    if state_id != state:
        return "Wrong state ID"
    return {
        "model": user_id in models,
        "dataset": user_id in datasets,
        "accuracy": user_id in accuracies,
    }


@app.post("/label")
def label(labels: Labels):
    user_id = labels.user_id
    state_id = labels.state_id
    if labels.state_id != state:
        return "Wrong state ID"
    for sample_id, label in labels.labels.items():
        datasets[user_id][int(sample_id)] = label
    acc_est = validate(state_id, user_id, [int(k) for k in labels.labels.keys()])
    if acc_est is not None:
        accuracies[user_id] *= 0.9
        accuracies[user_id] += 0.1 * acc_est["acc"]
    wtflabel.train.train(
        models[user_id],
        datasets[user_id],
        mnist,
        device=device,
    )
    print(accuracies[user_id])
    return {"accEst": accuracies[user_id]}


@app.get("/validate")
def validate(state_id:str, user_id: str, sample_ids: Optional[List[int]]):
    if state_id != state:
        return "Wrong state ID"
    acc_est = wtflabel.train.validate(
        models[user_id],
        datasets[user_id],
        mnist,
        sample_ids=sample_ids,
    )
    return {"acc": acc_est}


@app.get("/allUsers")
def get_all_users(state:str):
    if state_id != state:
        return "Wrong state ID"
    return [k for k in models.keys()]


app.mount("/mnist", StaticFiles(directory="mnist", html=True), name="mnist")


if __name__ == "__main__":
    from tqdm.auto import tqdm, trange
    user_id = init()["userId"]
    running_acc = 0.
    for i in trange(60):
        samples = get_samples(user_id)["samples"]
        labels = dict(zip(samples, np.array(real_labels)[samples]))
        label(labels, user_id)["accEst"]
        if i % 10 == 0:
            tqdm.write(f"Labeled {len(datasets[user_id])} samples...")
            tqdm.write("Estimated Accuracy: {:.2f}%".format(accuracies[user_id] * 100))
        
    print("Sampling 5 groups...")
    for _ in range(5):
        group = get_samples_group(user_id)
        samples = group["samples"]
        pred_label = group["predLabel"]
        label_dist = Counter(np.array(real_labels)[samples])
        print(f"Predicted label of {pred_label} for this group.")
        print(f"\tActual label distribution:\n\t{label_dist}")
        print("\tAccuracy ~ {:.2f}%".format(label_dist[pred_label] / len(samples) * 100))
