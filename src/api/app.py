import json
import math
from fastapi import FastAPI

MODEL_PATH = "models/model.json"

app = FastAPI()


@app.on_event("startup")
def load_model() -> None:
    global weights, bias
    with open(MODEL_PATH) as f:
        data = json.load(f)
    weights = data["weights"]
    bias = data["bias"]


@app.post("/predict")
def predict(payload):
    features = payload["features"]
    z = sum(w * f for w, f in zip(weights, features)) + bias
    prob = 1.0 / (1.0 + math.exp(-z))
    return {"prediction": 1 if prob >= 0.5 else 0}
