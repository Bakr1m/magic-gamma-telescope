# MAGIC Gamma-Telescope Classifier

**Baseline-comparison study | Tabular ML: data → baselines → tuned MLP → leakage analysis**

## Business Context

The MAGIC atmospheric Cherenkov telescope records flashes from particle
showers; most are hadron background, a minority are gamma signal. This
study asks which model family best separates the two on 10 engineered
image parameters — and, just as importantly, documents two methodology
bugs found in the original analysis and fixed here.

## Dataset

- **Source**: MAGIC04, UCI ML Repository (CC BY 4.0); reproduced via
  `scripts/download_data.sh` (data/ is gitignored)
- **Scale**: 19,020 rows — 12,332 gamma (64.8%) / 6,688 hadron (35.2%)
- **Features**: fLength, fWidth, fSize, fConc, fConc1, fAsym, fM3Long,
  fM3Trans, fAlpha, fDist
- **Split**: stratified 60/20/20, `random_state=42`

## Approach

1. **EDA** (`notebooks/01`): per-feature gamma/hadron histograms.
2. **Preprocessing** (`src/preprocessing.py`): StandardScaler fit on train
   only; train oversampled; val/test keep natural prevalence.
3. **Baselines** (`src/models.py`): KNN(5), GaussianNB, LogisticRegression, SVM.
4. **MLP grid** (`src/train.py`): 2×Dense + Dropout + sigmoid over 54
   configs, selection on validation loss, one final test report.
5. **Leakage analysis** (`tests/test_preprocessing.py`): regression tests
   pinning both fixes (see Methodology notes).

## Results (held-out test)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| MLP (grid best) | 0.88 | 0.86 |
| SVM | 0.86 | 0.85 |
| KNN(5) | 0.82 | 0.80 |
| Logistic Regression | 0.79 | 0.77 |
| Naive Bayes | 0.73 | 0.66 |

MLP per-class: hadron P/R 0.89/0.74, gamma P/R 0.87/0.95 — background
rejection is the weak side.

## Methodology Notes

The original notebook had two real bugs, kept visible in `notebooks/01`
and fixed in `src/`:
1. **Scaler leakage** — fresh `fit_transform` per split leaked val/test
   statistics into their own features.
2. **Unstratified split** — `np.split` on a global shuffle gave no class
   balance guarantee.

## Limitations

1. Simulated gammas + real hadrons — domain gap to live data.
2. Default-hyperparameter baselines (MLP grid excepted).
3. Selection on validation loss, single test report — no CV.
4. No calibration; engineered features only.

## Run Instructions

```bash
make install-dev            # runtime + train + pytest/ruff
make test lint              # 10 tests, ruff clean
bash scripts/download_data.sh  # fetch MAGIC04 into data/
make train                  # 54-config grid (~1h CPU); saves models/mlp_best.keras + grid_report.json
```
