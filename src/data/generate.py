from sklearn.datasets import make_classification


def load_data(n_samples: int = 1000, n_features: int = 20, random_state: int = 42):
    """Generate a synthetic imbalanced classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=2,
        n_redundant=2,
        n_classes=2,
        weights=[0.95, 0.05],
        random_state=random_state,
    )
    return X, y

