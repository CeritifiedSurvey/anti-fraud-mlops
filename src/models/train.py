import json
import math
import os
from pathlib import Path
from typing import List

from data.generate import load_data

MODEL_PATH = "models/model.joblib"


class LogisticRegressionModel:
    """A very small logistic regression implementation.

    The model is intentionally simple and relies only on the Python standard
    library so that no external dependencies are required.
    """

    def __init__(self, weights: List[float], bias: float) -> None:
        self.weights = weights
        self.bias = bias

    def predict(self, features: List[float]) -> int:
        linear = sum(w * x for w, x in zip(self.weights, features)) + self.bias
        prob = 1 / (1 + math.exp(-linear))
        return int(prob >= 0.5)


def train_model(model_path: str = MODEL_PATH) -> str:
    """Train a tiny logistic regression model and persist it.

    The training routine uses vanilla gradient descent and writes the resulting
    weights to ``model_path`` in JSON format (despite the ``.joblib``
    extension, which is kept for backwards compatibility with earlier
    versions).
    """

    X, y = load_data()
    n_features = len(X[0])
    weights = [0.0] * n_features
    bias = 0.0
    learning_rate = 0.1

    for _ in range(100):
        for features, label in zip(X, y):
            linear = sum(w * x for w, x in zip(weights, features)) + bias
            pred = 1 / (1 + math.exp(-linear))
            error = pred - label
            for j in range(n_features):
                weights[j] -= learning_rate * error * features[j]
            bias -= learning_rate * error

    model = {"weights": weights, "bias": bias}
    path = Path(model_path)
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f)
    return str(path)


__all__ = ["train_model", "MODEL_PATH", "LogisticRegressionModel"]
