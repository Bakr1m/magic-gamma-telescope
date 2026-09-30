"""Data tests: hermetic (synthetic same-schema fallback, no download)."""
import numpy as np
import pandas as pd

from src.data import DATA_PATH, FEATURE_COLS, TARGET_COL, load_magic, stratified_split


def _synthetic(n=600, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLS))), columns=FEATURE_COLS)
    df[TARGET_COL] = rng.choice([0, 1], n, p=[0.35, 0.65])
    return df


def _frame():
    if DATA_PATH.exists():
        return load_magic()
    return _synthetic()


def test_schema():
    df = _frame()
    assert list(df.columns) == FEATURE_COLS + [TARGET_COL]
    assert set(df[TARGET_COL].unique()) <= {0, 1}


def test_stratified_split_ratios_and_balance():
    df = _frame()
    train, valid, test = stratified_split(df)
    n = len(df)
    assert abs(len(train) / n - 0.60) < 0.03
    assert abs(len(valid) / n - 0.20) < 0.03
    assert abs(len(test) / n - 0.20) < 0.03
    overall = df[TARGET_COL].mean()
    for split in (train, valid, test):
        assert abs(split[TARGET_COL].mean() - overall) < 0.05


def test_stratified_split_reproducible():
    df = _frame()
    a = stratified_split(df)
    b = stratified_split(df)
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x, y)
