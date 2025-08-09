import json
import math
import os
from pathlib import Path
from typing import List

from data.generate import load_data

MODEL_PATH = "models/model.json"


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def train_model(model_path: str = MODEL_PATH) -> str:
    """Train a tiny logistic regression model using only the standard library.

    The model is fitted with stochastic gradient descent and the resulting
    weights are persisted as JSON.  This avoids external dependencies such as
    scikit-learn while keeping the example easy to follow.
    """
    X, y = load_data()
    n_features = len(X[0])
    weights: List[float] = [0.0] * n_features
    bias = 0.0
    lr = 0.1
    for _ in range(100):
        for features, label in zip(X, y):
            z = sum(w * f for w, f in zip(weights, features)) + bias
            pred = _sigmoid(z)
            error = pred - label
            for i in range(n_features):
                weights[i] -= lr * error * features[i]
            bias -= lr * error
    model_path_obj = Path(model_path)
    os.makedirs(model_path_obj.parent, exist_ok=True)
    with open(model_path_obj, "w") as f:
        json.dump({"weights": weights, "bias": bias}, f)
    return str(model_path_obj)
