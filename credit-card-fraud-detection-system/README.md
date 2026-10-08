# Credit Card Fraud Detection System

This project builds a fraud detection workflow for the IEEE-CIS credit card data. The focus is not only on getting a strong classifier, but on making the result reproducible and usable: the data is split in time order, every transformation is fitted on the correct partition, the decision threshold is treated as a business decision, and the complete raw-row pipeline is tracked as one model artifact.

The final workflow covers data preparation, feature engineering, GPU-capable feature selection, model comparison, calibration, frozen holdout evaluation, MLflow tracking, model registration, and batch scoring of new transactions.

## Dataset

The project uses the IEEE-CIS Fraud Detection dataset. It contains a transaction table and a separate identity table that are joined on `TransactionID`.

| Dataset part | Rows | Description |
|---|---:|---|
| Labeled transaction table | 590,540 | Transaction amounts, time, card, address, email, device and engineered `V` fields |
| Labeled identity table | 144,233 | Device and identity attributes, with substantial missingness |
| Unlabeled transaction table | 506,691 | Later transactions used for final batch scoring |
| Unlabeled identity table | 141,907 | Identity attributes for the later transactions |
| Fraud cases in the labeled transaction table | 20,663 | 3.50% of the labeled population |

The labeled transaction table has 394 columns, including `isFraud`; the identity table has 41 columns. After joining on `TransactionID`, the modeling table contains 434 raw columns before feature processing. The target is available only for the training period. The test period is later in `TransactionDT` and has no labels, so its output can be used for ranking and alert generation but cannot be scored with precision or recall until fraud outcomes arrive.

The data is difficult for three practical reasons:

1. Fraud is rare, so accuracy would hide poor fraud detection.
2. Many identity and device fields are sparse, high-cardinality, or categorical.
3. Transactions have time order. A random split would allow information from the future to influence the past and would give an overly optimistic estimate.

Raw data is deliberately not stored in this repository. Download the genuine IEEE-CIS files and place them under `data/raw/` before running the notebooks.

## System architecture

The same raw-row contract is carried through training, tracking, and inference. The model receives a joined transaction row, applies the fitted transformations, returns a probability, and applies the stored alert threshold. The four layers below show how a transaction moves from raw data to a monitored deployment.

```
flowchart TB
    subgraph DATA["1 · DATA ENGINEERING"]
        direction LR
        A["IEEE-CIS Raw Data"] --> B["Validation + Bronze / Silver / Gold"]
        B --> C["Chronological Split + Data Versioning"]
    end

    subgraph MODEL["2 · MODEL DEVELOPMENT"]
        direction LR
        D["Feature Engineering + Selection"] --> E["Train + Compare Models"]
        E --> F["Evaluation + Quality Gate"]
    end

    subgraph REGISTRY["3 · TRACKING & MODEL REGISTRY"]
        direction LR
        G["MLflow Experiment Tracking"] --> H["Register Approved Model"]
    end

    subgraph DEPLOY["4 · TESTING & DEPLOYMENT"]
        direction LR
        I["GitHub Actions + Tests"] --> J["Docker + Azure ML Endpoint"]
    end

    subgraph MONITOR["5 · MONITORING & CONTINUOUS IMPROVEMENT"]
        direction LR
        K["Predictions + Logs"] --> L["Drift + Performance + Alerts"]
        L --> M["Retraining Decision"]
    end

    C --> D
    E -. "Log experiments" .-> G
    F --> H
    H --> I
    J --> K
    M -. "Fresh validated data" .-> A

    classDef data fill:#E8F1FF,stroke:#5B8DEF,color:#173B72;
    classDef model fill:#F0E9FF,stroke:#8B6CE8,color:#3E2D70;
    classDef registry fill:#E7F5FF,stroke:#3598CB,color:#145678;
    classDef deploy fill:#FFF3D6,stroke:#E4A62A,color:#704A00;
    classDef ops fill:#E5F8EB,stroke:#4DBD73,color:#145C2C;

    class A,B,C data;
    class D,E,F model;
    class G,H registry;
    class I,J deploy;
    class K,L,M ops;
```

The local notebook workflow implements the steps through tracking, registration, and batch inference. The registered artifact is the deployment unit for the Azure serving layer; the serving wrapper does not refit the model or learn new encodings at request time.

| Layer | Responsibility | Main components |
|---|---|---|
| Data & exploration | Validate source files, join tables, profile quality, and preserve time order | Pandas, manifests, training-only EDA |
| Model development | Fit transformations, select features, compare models, calibrate probabilities, and gate on holdout | scikit-learn, LightGBM, XGBoost, optional GPU acceleration |
| Deployment | Track the artifact, register the approved version, and expose batch or online scoring | MLflow, Docker, Azure Container Registry, Azure ML |
| Operations | Watch service health and model behavior, then start a controlled retraining cycle | Azure Monitor, Application Insights, delayed fraud labels |

## Machine-learning pipeline

### 1. Data validation and reproducible boundaries

The preparation stage checks that the expected files are real data rather than missing files or Git LFS pointers. It validates identifiers, joins the transaction and identity tables, records source fingerprints, and creates chronological partitions. The split is decided before model fitting and is kept unchanged through the later notebooks.

### 2. Feature engineering

The feature pipeline keeps raw values and creates groups of transaction, card, address, email, device, identity, time, missingness, and `V`-family features. Categorical values are encoded with fitted mappings, numerical values are imputed inside the fitted pipeline, and optional clustering features are learned only from training data. The same fitted pipeline is reused for validation, holdout, and the unlabeled test table.

### 3. Feature selection

Several selectors are compared rather than trusting one ranking: target correlation, random-forest importance, L1 regularization, permutation or gradient-boosting rankings, and a consensus recipe. GPU LightGBM correlation/ranking support is available for the expensive numeric screening step. The selected feature list is stored with the model recipe so a later prediction cannot silently use a different schema.

### 4. Model comparison

The workflow compares linear and tree-based candidates, including logistic regression, random forest, XGBoost, and LightGBM. Average precision is the primary selection metric because the positive class is rare; ROC-AUC, precision, recall, F1, calibration, and alert volume are reported alongside it. SMOTE is not used because synthetic future-like rows would complicate the chronological evaluation and change the natural fraud prior. The workflow instead keeps the observed class distribution and tunes the operating threshold explicitly.

### 5. Calibration and alert policy

The selected LightGBM development model is calibrated on a separate validation block. The threshold is selected against validation F1 and is stored inside the complete `FraudPipeline`. This means a prediction service returns both the probability and the same alert label that was evaluated during model selection.

### 6. Frozen holdout assessment

After the winner and threshold are frozen, the holdout is evaluated once. It is not used to choose features, tune hyperparameters, or change the threshold. This keeps the holdout as the final unbiased development checkpoint.

## Results

The table below separates the validation improvement experiment from the frozen holdout assessment. The calibrated LightGBM row is a development comparison on validation data; the holdout row is kept as the final frozen assessment of the selected pipeline.

| Model / assessment | Split | ROC-AUC | Average precision | Precision | Recall | F1 | Brier score |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline pipeline | Validation | 0.9042 | 0.4688 | 0.5897 | 0.3602 | 0.4473 | 0.0939 |
| Calibrated GPU LightGBM | Validation | **0.9071** | **0.5192** | **0.6456** | **0.4301** | **0.5163** | **0.0232** |
| Frozen selected pipeline | Holdout | 0.8815 | 0.4524 | 0.5042 | 0.4122 | 0.4536 | 0.1021 |

The validation experiment improved average precision, F1, and probability calibration while keeping a transparent threshold policy. The holdout remains a separate frozen result; a new calibrated model should receive a new holdout assessment before it is promoted as the production candidate.

### Unlabeled test scoring

The complete raw-row inference path was run on all **506,691** unlabeled transactions.

| Output | Result |
|---|---:|
| Rows scored | 506,691 |
| Alert threshold | 0.2019 |
| Fraud alerts | 15,848 |
| Alert rate | 3.13% |

The unlabeled table has no ground-truth target, so this run verifies row coverage, ID order, probability range, and alert volume. Precision and recall will be measured later when confirmed fraud outcomes are joined back to the scored transactions.

## MLOps lifecycle

### Experiment tracking and registry

MLflow records the comparison run and the final candidate in the `ieee_cis_fraud_detection` experiment. Each final run stores the fitted raw-row pipeline, metrics, model parameters, source fingerprints, partition fingerprints, input signature, and supporting reports. The registered model name is `ieee_cis_fraud_detector`, and a deployment can resolve a version with:

```text
models:/ieee_cis_fraud_detector/<version>
```

The registry verification step reloads the registered artifact, checks that its lineage matches the current partitions, and confirms that its probabilities and labels match the locally saved pipeline.

### Deployment path

The production deployment uses the registered artifact as the single source of truth:

1. Package the prediction wrapper with the model-loading and raw-row schema checks.
2. Build a small container for the API and install the pinned project dependencies.
3. Push the image to Azure Container Registry.
4. Deploy the image as an Azure Machine Learning managed online endpoint (or an Azure Container App when a lighter HTTP service is preferred).
5. Keep the model URI, threshold, and schema in deployment configuration rather than hard-coding a second copy of the model.
6. Expose health, prediction, latency, error, and alert-volume signals to Azure Monitor and Application Insights.

### Monitoring and retraining

The monitoring loop tracks request health and model behavior separately. It checks missingness, unseen categories, numeric ranges, input drift, prediction distributions, alert rate, and response latency. When confirmed fraud labels arrive, it adds precision, recall, average precision, calibration, and false-positive cost to the monitoring report. A scheduled retraining job can then create a new candidate, repeat the chronological evaluation, log it to MLflow, and promote it only after the new holdout and lineage gates pass.

## Repository layout

```text
credit-card-fraud-detection-system/
├── eda/
│   └── 01_explore_transactions.ipynb
├── notebook/
│   ├── 00_data_preparation.ipynb
│   ├── 01_feature_engineering.ipynb
│   ├── 02_feature_selection.ipynb
│   ├── 03_modeling_baselines.ipynb
│   └── 04_model_selection.ipynb
├── src/fraud_detection/
│   ├── data.py
│   ├── data_preparation.py
│   ├── features.py
│   ├── feature_selection.py
│   ├── models.py
│   ├── experiments.py
│   ├── evaluation.py
│   ├── predict.py
│   └── tracking.py
├── requirements.txt
├── requirements-local.txt
├── requirements-gpu.txt
└── pyproject.toml
```

Generated data, model binaries, prediction exports, MLflow databases, credentials, and notebook caches are ignored so the repository remains reviewable and reproducible.

## Run locally

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

Place the genuine IEEE-CIS files under `data/raw/`, then run notebooks `00` through `04` in order. To turn on tracking in notebook `04`:

```python
import os

os.environ["FRAUD_NOTEBOOK_TRACK"] = "1"
os.environ["FRAUD_NOTEBOOK_REGISTER"] = "1"
```

For GPU experiments, install `requirements-gpu.txt` in a compatible CUDA environment. For the local MLflow dashboard, use the tracking URI produced by the notebook and open the resulting local server in a browser.

## Scope and data responsibility

This repository contains the executable workflow and its documentation, not the competition data or a production customer feed. Model quality should be rechecked on the target institution's data, with an agreed false-positive budget, fraud-label delay, access controls, and retention policy before a live decision service is used.
