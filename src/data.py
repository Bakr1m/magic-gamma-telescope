"""MAGIC gamma-telescope data: load + stratified split.

Dataset: MAGIC04 (UCI ML Repository; Monte-Carlo gamma showers registered
by the MAGIC atmospheric Cherenkov telescope, La Palma). 19,020 rows,
10 numeric image parameters, label g (gamma, signal) / h (hadron, bg).
CC BY 4.0.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "magic04.data"

FEATURE_COLS = [
    "fLength", "fWidth", "fSize", "fConc", "fConc1",
    "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist",
]
TARGET_COL = "class"

TRAIN_RATIO = 0.60
VALID_RATIO = 0.20
TEST_RATIO = 0.20
SEED = 42


def load_magic(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load raw MAGIC04; encode label g->1 (signal), h->0 (background)."""
    df = pd.read_csv(path, names=FEATURE_COLS + [TARGET_COL])
    df[TARGET_COL] = (df[TARGET_COL] == "g").astype(int)
    return df


def stratified_split(df: pd.DataFrame, seed: int = SEED):
    """Stratified 60/20/20 split, reproducible.

    The notebook used unstratified np.split on a global shuffle (no class
    balance guarantee, seed from global RNG state). This is the fix.
    """
    train, valid_test = train_test_split(
        df, test_size=VALID_RATIO + TEST_RATIO, stratify=df[TARGET_COL], random_state=seed
    )
    valid, test = train_test_split(
        valid_test,
        test_size=TEST_RATIO / (VALID_RATIO + TEST_RATIO),
        stratify=valid_test[TARGET_COL],
        random_state=seed,
    )
    return (
        train.reset_index(drop=True),
        valid.reset_index(drop=True),
        test.reset_index(drop=True),
    )
