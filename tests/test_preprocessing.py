"""Preprocessing tests: the scaler-leakage regression test.

The notebook fit a fresh StandardScaler per split. These tests pin the
fix: one scaler, fit on train, applied everywhere.
"""
import numpy as np
import pytest

from src.preprocessing import oversample_train, scale_splits


@pytest.fixture()
def splits():
    rng = np.random.default_rng(7)
    X_train = rng.normal(loc=0.0, scale=1.0, size=(200, 10))
    X_valid = rng.normal(loc=5.0, scale=2.0, size=(60, 10))  # shifted on purpose
    X_test = rng.normal(loc=-3.0, scale=0.5, size=(60, 10))
    return X_train, X_valid, X_test


def test_scaler_uses_train_statistics_only(splits):
    X_train, X_valid, X_test = splits
    scaler, Xs_train, Xs_valid, Xs_test = scale_splits(X_train, X_valid, X_test)
    np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0), rtol=1e-10)
    # Manual check: every split standardized with TRAIN mean/var.
    for X, Xs in ((X_train, Xs_train), (X_valid, Xs_valid), (X_test, Xs_test)):
        np.testing.assert_allclose(
            Xs, (X - X_train.mean(axis=0)) / X_train.std(axis=0), rtol=1e-10
        )


def test_per_split_fit_transform_gives_different_results(splits):
    """Documents the notebook bug: independent fit_transform != shared fit."""
    from sklearn.preprocessing import StandardScaler

    X_train, X_valid, _ = splits
    shared = scale_splits(X_train, X_valid, X_valid)[2]
    independent = StandardScaler().fit_transform(X_valid)
    assert not np.allclose(shared, independent)


def test_train_is_standardized(splits):
    X_train, _, _ = splits
    _, Xs_train, _, _ = scale_splits(X_train, X_train, X_train)
    np.testing.assert_allclose(Xs_train.mean(axis=0), 0.0, atol=1e-10)


def test_oversample_balances_train_only():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    y = np.array([0] * 80 + [1] * 20)
    Xo, yo = oversample_train(X, y)
    assert len(Xo) > len(X)
    assert (yo == 0).sum() == (yo == 1).sum()
