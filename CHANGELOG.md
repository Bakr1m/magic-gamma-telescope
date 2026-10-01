# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2026-09-30

### Added
- Refactored study notebook into `src/`: `data` (MAGIC04 loader +
  stratified 60/20/20 split), `preprocessing` (fit-on-train scaler +
  train-only oversampling), `models` (thesis MLP + 4 classical baselines),
  `train` (54-config grid search as a function, JSON report).
- Hermetic pytest suite (schema, stratification, scaler-leakage regression
  test, architecture fidelity, training smoke test); ruff lint; CI.
- Professional repo hygiene: CONTRIBUTING, CHANGELOG, CI workflow,
  Makefile, model card, raw data removed from tracking + download script.

### Fixed
- **Scaler leakage**: fresh `fit_transform` per split replaced with
  fit-on-train/transform-everywhere (pinned by regression test).
- **Unstratified split**: `np.split` on global shuffle replaced with
  stratified `train_test_split(random_state=42)`.
- Raw `magic04.data` (1.4 MB) removed from git; reproduced via
  `scripts/download_data.sh`.
