# Anti-Fraud MLOps

This repository demonstrates a tiny, self-contained example of an anti-fraud
machine learning workflow.  Synthetic data is generated, a logistic regression
model is trained using only the Python standard library, and a lightweight API
is provided for prediction.

## Setup

No third-party packages are required.  Installing the requirements file will not
fetch anything:

```bash
pip install -r requirements.txt
```

## Training

Train the model and persist it to `models/model.json`:

```bash
python -m models.train
```

## API

A minimal FastAPI-compatible framework is included solely for testing.  Example
usage:

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
