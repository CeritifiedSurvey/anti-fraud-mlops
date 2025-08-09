# Anti-Fraud MLOps

This repository demonstrates a tiny, self-contained example of an anti-fraud
machine learning workflow. Synthetic data is generated, a logistic regression
model is trained using only the Python standard library, and a lightweight
FastAPI-compatible service exposes predictions.

## Setup

Install the (empty) dependencies list to mirror a typical workflow:

```bash
pip install -r requirements.txt
```

## Training

Train the model and persist it to `models/model.joblib`:

```bash
python -m models.train
```

## API

The trained model can be served via the included FastAPI-compatible stub.
Example usage with the provided endpoint:

```python
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)
response = client.post("/predict", json={"features": [0.0] * 20})
print(response.json())
```

## Testing

Run the unit tests with:

```bash
pytest
```
