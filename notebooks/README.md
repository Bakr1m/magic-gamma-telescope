# Notebooks — exploratory trail (docs, not the deploy path)

- `01_magic_baselines.ipynb` — original study notebook: MAGIC04 EDA
  (per-feature gamma/hadron histograms), KNN / Naive Bayes / Logistic
  Regression / SVM baselines, and a 54-config MLP grid search. The
  notebook's `scale_dataset` (fit-transform per split) and unstratified
  `np.split` are preserved as history; the corrected, reproducible path
  is `src/` + `make train`. See `docs/model_card.md` for the analysis.
