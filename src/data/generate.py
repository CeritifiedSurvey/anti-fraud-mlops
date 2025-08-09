import random
from typing import List, Tuple


def load_data(
    n_samples: int = 200, n_features: int = 20, random_state: int = 42
) -> Tuple[List[List[float]], List[int]]:
    """Generate a simple synthetic dataset using only the standard library.

    Each sample consists of ``n_features`` random floats in ``[0, 1)``.  The
    label is determined by summing the feature values and adding a small amount
    of noise.  Points with a sufficiently large sum are labelled as fraud
    (``1``), while the rest are non-fraud (``0``).
    """
    random.seed(random_state)
    X: List[List[float]] = []
    y: List[int] = []
    for _ in range(n_samples):
        features = [random.random() for _ in range(n_features)]
        noise = random.uniform(-0.5, 0.5)
        label = 1 if sum(features) + noise > n_features * 0.6 else 0
        X.append(features)
        y.append(label)
    return X, y
