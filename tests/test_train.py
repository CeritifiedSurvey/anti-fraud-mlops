from models.train import train_model


def test_train_creates_model(tmp_path):
    model_path = tmp_path / "model.joblib"
    train_model(model_path=str(model_path))
    assert model_path.exists()

