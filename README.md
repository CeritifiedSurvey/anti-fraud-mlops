# Anti-Fraud MLOps

This repository provides a minimal end-to-end example of an anti-fraud machine learning pipeline including data generation, model training, and a simple prediction API.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

```bash
python -m models.train
```

This generates a logistic regression model stored at `models/model.joblib`.

## API

After training, start the API:

```bash
uvicorn api.app:app --reload
```

Send a POST request to `/predict` with JSON payload containing a `features` list of floats.

## Testing

Run tests with:

```bash
pytest
```

