import json
import math

from fastapi import FastAPI

MODEL_PATH = "models/model.joblib"

app = FastAPI()


class Model:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def predict(self, features):
        linear = sum(w * x for w, x in zip(self.weights, features)) + self.bias
        prob = 1 / (1 + math.exp(-linear))
        return int(prob >= 0.5)


@app.on_event("startup")
def load_model() -> None:
    """Load the persisted model into memory when the app starts."""
    global model
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    model = Model(data["weights"], data["bias"])


@app.post("/predict")
def predict(payload: dict) -> dict:
    """Return a fraud prediction for the provided feature vector."""
    pred = model.predict(payload["features"])
    return {"prediction": pred}
