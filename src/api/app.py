from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

MODEL_PATH = "models/model.joblib"

app = FastAPI()


class Features(BaseModel):
    features: list[float]


@app.on_event("startup")
def load_model():
    global model
    model = joblib.load(MODEL_PATH)


@app.post("/predict")
def predict(features: Features):
    data = np.array(features.features).reshape(1, -1)
    pred = model.predict(data)[0]
    return {"prediction": int(pred)}

