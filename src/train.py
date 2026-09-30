"""Grid search as a function: MLP hyperparameter sweep + test report.

Thesis grid: num_nodes in {16, 32, 64}, dropout in {0, 0.2},
lr in {0.01, 0.005, 0.001}, batch_size in {32, 64, 128} (54 configs,
100 epochs). Selection on validation loss; final report on held-out test.
"""
import itertools
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path

from sklearn.metrics import classification_report

from src.data import TARGET_COL, load_magic, stratified_split
from src.models import build_baseline, train_mlp
from src.preprocessing import oversample_train, scale_splits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = PROJECT_ROOT / "models" / "grid_report.json"

GRID = {
    "num_nodes": [16, 32, 64],
    "dropout_prob": [0.0, 0.2],
    "lr": [0.01, 0.005, 0.001],
    "batch_size": [32, 64, 128],
}
EPOCHS = 100


def prepare_arrays(df_train, df_valid, df_test):
    feature_cols = [c for c in df_train.columns if c != TARGET_COL]
    splits = [
        (df[feature_cols].values, df[TARGET_COL].values) for df in (df_train, df_valid, df_test)
    ]
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = splits
    _, X_train_s, X_valid_s, X_test_s = scale_splits(X_train, X_valid, X_test)
    X_train_s, y_train = oversample_train(X_train_s, y_train)
    return (X_train_s, y_train), (X_valid_s, y_valid), (X_test_s, y_test)


def run_grid(epochs: int = EPOCHS, grid: dict = GRID):
    df = load_magic()
    train_df, valid_df, test_df = stratified_split(df)
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = prepare_arrays(
        train_df, valid_df, test_df
    )

    results, best = [], {"val_loss": float("inf")}
    for num_nodes, dropout_prob, lr, batch_size in itertools.product(
        grid["num_nodes"], grid["dropout_prob"], grid["lr"], grid["batch_size"]
    ):
        model, _ = train_mlp(
            X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs
        )
        val_loss, val_acc = model.evaluate(X_valid, y_valid, verbose=0)
        entry = {
            "num_nodes": num_nodes, "dropout_prob": dropout_prob,
            "lr": lr, "batch_size": batch_size,
            "val_loss": float(val_loss), "val_acc": float(val_acc),
        }
        results.append(entry)
        if val_loss < best["val_loss"]:
            best = {**entry, "model": model}
        print(f"nodes={num_nodes} drop={dropout_prob} lr={lr} bs={batch_size} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    y_pred = (best["model"].predict(X_test, verbose=0) > 0.5).astype(int).ravel()
    report = classification_report(y_test, y_pred, digits=4, output_dict=False)
    print(report)

    baselines = {}
    for name in ("knn", "naive_bayes", "logistic_regression", "svm"):
        clf = build_baseline(name).fit(X_train, y_train)
        baselines[name] = float(clf.score(X_test, y_test))
    print("baselines:", baselines)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(
        {"best": {k: v for k, v in best.items() if k != "model"},
         "test_report": report, "baselines": baselines, "n_configs": len(results)}, indent=2))
    best["model"].save(PROJECT_ROOT / "models" / "mlp_best.keras")
    return best, report


if __name__ == "__main__":
    run_grid()
