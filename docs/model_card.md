# Model Card — MAGIC Gamma/Hadron Classifier

## Intended use
Baseline-comparison study on a classic astroparticle dataset: which model
family separates gamma showers from hadron background on 10 Cherenkov
image parameters. Research artifact — not a detector component.

## Data
- **Source**: MAGIC04, UCI ML Repository (Monte-Carlo gamma events +
  real hadron background registered by the MAGIC telescope; CC BY 4.0).
- **Scale**: 19,020 rows; 12,332 gamma (64.8%) / 6,688 hadron (35.2%).
- **Features**: 10 Hillas-style image parameters (fLength, fWidth, fSize,
  fConc, fConc1, fAsym, fM3Long, fM3Trans, fAlpha, fDist).
- **Split**: stratified 60/20/20 (`random_state=42`).

## Method
- Preprocessing: StandardScaler **fit on train only**; train oversampled
  (RandomOverSampler) to balance classes; val/test keep natural prevalence.
- MLP grid: 2×Dense(n) + Dropout + sigmoid; n ∈ {16,32,64}, dropout ∈
  {0,0.2}, lr ∈ {0.01,0.005,0.001}, batch ∈ {32,64,128} — 54 configs,
  100 epochs, selection on validation loss.
- Baselines: KNN(5), GaussianNB, LogisticRegression, SVM (defaults).

## Metrics (held-out test, corrected pipeline rerun)
| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| MLP (64 nodes, dropout 0.2, lr 0.001, batch 32) | 0.8754 | 0.8571 |
| SVM | 0.8570 | — |
| KNN(5) | 0.8128 | — |
| Logistic Regression | 0.7789 | — |
| Naive Bayes | 0.7229 | — |

MLP detail: hadron P/R 0.8906/0.7362, gamma P/R 0.8692/0.9509 —
background rejection is the weak side. (Notebook reported 0.88/0.86
under the leaky scaler; corrected rerun confirms the ranking.)

## Methodology notes (fixed in `src/`, preserved in the notebook)
1. **Scaler leakage**: the notebook fit a fresh scaler per split; `src/`
   fits once on train (regression-tested).
2. **Unstratified split**: the notebook used `np.split` on a global
   shuffle; `src/` stratifies (balance-tested).
3. **Selection protocol**: grid selects on validation loss and reports on
   test once — documented, not hidden.

## Limitations
- Simulated gammas + real hadrons: domain gap to live telescope data.
- Default-hyperparameter baselines (except the MLP grid).
- No calibration; single split, no CV.
- 10 engineered features only — raw pixel data unused.

## Ethics
Fundamental-science dataset; no personal data, no dual-use concern. Demo only.
