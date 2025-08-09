from fastapi.testclient import TestClient
from models.train import train_model
from api.app import app


def setup_module(module):
    # Ensure a model exists for API tests
    train_model()


def test_predict_endpoint():
    client = TestClient(app)
    response = client.post("/predict", json={"features": [0.0] * 20})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], int)

