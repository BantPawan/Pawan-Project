# Credit Card Fraud Detection System

An IEEE-CIS fraud detection project built as a chronological machine-learning workflow. It combines feature engineering, GPU-capable feature selection, model comparison, threshold calibration, reproducible evaluation, MLflow tracking, and batch prediction.

## What is included

- `notebook/00` to `notebook/04`: preparation, feature engineering, selection, baseline modeling, and final model study.
- `src/fraud_detection`: reusable data, feature, model, evaluation, prediction, and MLflow code.
- `eda/01_explore_transactions.ipynb`: exploratory analysis.
- `requirements.txt`: complete project environment.
- `requirements-local.txt`: smaller local workflow environment.
- `requirements-gpu.txt`: optional CUDA acceleration dependencies.

Raw IEEE-CIS data, generated reports, local models, MLflow databases, and prediction exports are intentionally excluded. Place the genuine transaction and identity files under `data/raw/` before running the notebooks.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

For GPU experiments, install `requirements-gpu.txt` and use a LightGBM GPU-enabled environment.

## Workflow

Run notebooks `00` through `04` in order. The final notebook freezes the selected model, evaluates the untouched holdout, creates the complete raw-row prediction pipeline, logs the comparison and final model to MLflow, registers the model, and scores unlabeled test rows.

To enable MLflow in notebook `04`, set these variables before running its tracking cell:

```python
import os
os.environ["FRAUD_NOTEBOOK_TRACK"] = "1"
os.environ["FRAUD_NOTEBOOK_REGISTER"] = "1"
```

The model is registered as `ieee_cis_fraud_detector` and can be loaded through MLflow with `models:/ieee_cis_fraud_detector/<version>`.

## Current development result

The improved development evaluation achieved approximately **0.907 ROC-AUC**, **0.519 average precision**, **0.516 F1**, and **0.023 Brier score** on its validation comparison. The unlabeled test pipeline successfully scored **506,691 rows** and produced **15,848 alerts** at the calibrated threshold. These figures are development evidence; production performance requires delayed fraud labels and monitoring.
