"""Scaling: fit on train, transform val/test (no leakage).

The notebook's scale_dataset called fit_transform on every split
independently — each split got its own mean/variance, so (a) val/test
statistics leaked into their own features and (b) identical showers were
encoded differently per split. This module fits once on train.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def scale_splits(X_train, X_valid, X_test):
    """Scale all splits with train statistics. Val/test keep natural prevalence."""
    scaler = fit_scaler(np.asarray(X_train, dtype=float))
    return (
        scaler,
        scaler.transform(np.asarray(X_train, dtype=float)),
        scaler.transform(np.asarray(X_valid, dtype=float)),
        scaler.transform(np.asarray(X_test, dtype=float)),
    )


def oversample_train(X_train, y_train, seed: int = 42):
    """Balance the train split only (thesis behavior), after scaling."""
    from imblearn.over_sampling import RandomOverSampler

    ros = RandomOverSampler(random_state=seed)
    return ros.fit_resample(X_train, y_train)
