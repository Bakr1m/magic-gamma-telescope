"""Model tests: architecture fidelity + training smoke test."""
import numpy as np
import pytest

from src.models import build_baseline, build_mlp, train_mlp


def test_mlp_matches_thesis_architecture():
    model = build_mlp(32, 0.2, 0.005, n_features=10)
    dense = [layer for layer in model.layers if "dense" in layer.name]
    assert [layer.units for layer in dense] == [32, 32, 1]
    assert dense[-1].activation.__name__ == "sigmoid"
    assert model.loss == "binary_crossentropy"


def test_baselines_construct():
    for name in ("knn", "naive_bayes", "logistic_regression", "svm"):
        assert build_baseline(name) is not None
    with pytest.raises(ValueError):
        build_baseline("xgboost")


def test_train_mlp_smoke_on_synthetic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 10)).astype("float32")
    y = (X[:, 0] > 0).astype("float32")
    model, history = train_mlp(X, y, 8, 0.0, 0.01, 16, epochs=2)
    assert "loss" in history.history
    assert model.predict(X[:4], verbose=0).shape == (4, 1)
