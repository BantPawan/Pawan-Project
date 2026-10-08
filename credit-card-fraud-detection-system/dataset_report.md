# Dataset Analysis Report
### IEEE-CIS Credit Card Fraud Detection — Supervisor Review
**Prepared by:** Antigravity AI  
**Date:** 08 October 2026  
**Project:** `credit-card-fraud-detection-system`  
**Data Location:** `data/raw/`

---

## 1. Overview

This project uses the **IEEE-CIS Fraud Detection** dataset (originally from Kaggle, 2019), provided by Vesta Corporation. The dataset captures real-world e-commerce transactions and is split into **transaction** and **identity** tables, further divided into training and test sets. The prediction task is a **binary classification**: identify whether a given transaction is fraudulent (`isFraud = 1`) or legitimate (`isFraud = 0`).

| File | Rows | Columns | Size on Disk | In-Memory |
|---|---|---|---|---|
| `train_transaction.csv` | 590,540 | 394 | ~652 MB | ~2,203 MB |
| `train_identity.csv` | 144,233 | 41 | ~25.3 MB | ~165 MB |
| `test_transaction.csv` | 506,691 | 393 | ~585 MB | ~1,896 MB |
| `test_identity.csv` | 141,907 | 41 | ~24.6 MB | ~162 MB |
| `sample_submission.csv` | 506,691 | 2 | ~5.8 MB | ~8 MB |

> [!NOTE]
> The **identity tables** cover only a subset of transactions (~24% of training rows). The join key is `TransactionID`. Not every transaction has a corresponding identity record — this is expected and must be handled carefully during feature engineering.

---

## 2. File-by-File Analysis

---

### 2.1 `train_transaction.csv` — Primary Training Table

**Shape:** 590,540 rows × 394 columns  
**Target variable:** `isFraud` (binary, 0 or 1)

#### 2.1.1 Target Variable — Class Imbalance

| Class | Count | Proportion |
|---|---|---|
| Legitimate (`isFraud = 0`) | 569,877 | **96.50%** |
| Fraudulent (`isFraud = 1`) | 20,663 | **3.50%** |

> [!CAUTION]
> The dataset is **severely imbalanced** — only 3.5% of transactions are fraudulent. Naïve classifiers will achieve ~96.5% accuracy by predicting all-legitimate, which is meaningless. You **must** use class-weighted loss functions, oversampling (SMOTE), undersampling, or a combination thereof. Evaluation must focus on **AUC-ROC, Precision-Recall AUC, and F1 score**, not raw accuracy.

#### 2.1.2 Transaction Timestamp

- **Column:** `TransactionDT` (seconds elapsed since a reference datetime)
- **Min:** 86,400 s (Day 1) | **Max:** 15,811,131 s
- **Span:** ~182 days (~6 months of training data)
- The test set's `TransactionDT` begins immediately after the training set ends, simulating a real deployment scenario.

> [!IMPORTANT]
> This is a **time-series split** problem. You must **not** use random train/validation splits. Always split chronologically to prevent data leakage. Temporal cross-validation (e.g., rolling-window) is the correct approach.

#### 2.1.3 Transaction Amount

| Statistic | Value |
|---|---|
| Mean | \$135.03 |
| Median | \$68.77 |
| Std Dev | \$239.16 |
| Min | \$0.25 |
| Max | \$31,937.39 |
| 25th Percentile | \$43.32 |
| 75th Percentile | \$125.00 |

The distribution is **heavily right-skewed** — the majority of transactions are under \$125, but there are extreme outliers above \$10,000. Log-transformation (`log1p`) of `TransactionAmt` is strongly recommended.

#### 2.1.4 Feature Groups

The 394 columns fall into the following semantic groups:

| Group | Columns | Count | Description |
|---|---|---|---|
| Core | `TransactionID`, `TransactionDT`, `TransactionAmt`, `isFraud` | 4 | Identifiers and target |
| Product | `ProductCD` | 1 | Product category (W, C, R, H, S) |
| Card | `card1–card6` | 6 | Card metadata (issuer, type, network) |
| Address | `addr1`, `addr2` | 2 | Billing address codes |
| Distance | `dist1`, `dist2` | 2 | Distance features (from identity or billing info) |
| Email | `P_emaildomain`, `R_emaildomain` | 2 | Purchaser and recipient email domains |
| Counting | `C1–C14` | 14 | Count-based features (how many addresses, cards, etc. associated with user) |
| Time-delta | `D1–D15` | 15 | Days since various events (last transaction, account creation, etc.) |
| Match flags | `M1–M9` | 9 | Binary match flags (name, address, zip, email, etc.) |
| Vesta features | `V1–V339` | 339 | Proprietary engineered features from Vesta |

#### 2.1.5 Product Code Distribution

| Code | Count | % Share |
|---|---|---|
| W | 439,670 | 74.5% |
| C | 68,519 | 11.6% |
| R | 37,699 | 6.4% |
| H | 33,024 | 5.6% |
| S | 11,628 | 2.0% |

Product `W` dominates the dataset. Consider whether fraud rates differ significantly across product codes — this is likely to be a valuable stratification feature.

#### 2.1.6 Card Network & Type

**Card Network (`card4`):**
| Network | Count |
|---|---|
| Visa | 384,767 (65.2%) |
| Mastercard | 189,217 (32.0%) |
| American Express | 8,328 (1.4%) |
| Discover | 6,651 (1.1%) |

**Card Type (`card6`):**
| Type | Count |
|---|---|
| Debit | 439,938 (74.5%) |
| Credit | 148,986 (25.2%) |
| Debit or Credit | 30 |
| Charge Card | 15 |

#### 2.1.7 Email Domains

Top purchaser (`P_emaildomain`) domains:
| Domain | Count |
|---|---|
| gmail.com | 228,355 |
| yahoo.com | 100,934 |
| hotmail.com | 45,250 |
| anonymous.com | 36,998 |
| aol.com | 28,289 |

> [!NOTE]
> `anonymous.com` appearing as the 4th most common domain is suspicious and warrants investigation — transactions using this domain likely have elevated fraud rates. Email domain engineering (grouping by TLD, flagging anonymous/disposable domains) is a high-value feature engineering step.

#### 2.1.8 Missing Data

- **Total missing values:** 95,566,686
- **Columns with >50% missing:** 174 out of 394

**Worst offenders (>80% missing):**
| Column | Missing % |
|---|---|
| `dist2` | 93.6% |
| `D7` | 93.4% |
| `D13` | 89.5% |
| `D14` | 89.5% |
| `D12` | 89.0% |
| `D6` | 87.6% |
| `D8`, `D9` | 87.3% |
| V153–V163 group | 86.1% |

> [!WARNING]
> **174 features have more than 50% missing values.** Many of the V-features form correlated subgroups — a large block is simultaneously populated or simultaneously missing. Simple median/mean imputation will be misleading. Consider: (a) null-indicator flags, (b) group-level imputation, or (c) dropping columns above a missingness threshold (e.g., >95%).

---

### 2.2 `train_identity.csv` — Identity Supplement (Training)

**Shape:** 144,233 rows × 41 columns  
**Coverage:** 24.4% of training transactions have identity records  
**Join key:** `TransactionID`

#### 2.2.1 Column Structure

| Group | Columns | Description |
|---|---|---|
| Numeric IDs | `id_01` – `id_11` | Numeric identity features (scores, offsets) |
| Categorical IDs | `id_12` – `id_38` | Categorical match/found flags, OS info |
| Device | `DeviceType`, `DeviceInfo` | Device category and raw device string |

**Data types:** 23 float64, 17 object, 1 int64

#### 2.2.2 Device Type Distribution

| Device Type | Count | % |
|---|---|---|
| Desktop | 85,165 | 59.0% |
| Mobile | 55,645 | 38.6% |

Desktop is slightly more prevalent in identity-linked transactions. Mobile transactions may have distinct fraud patterns.

#### 2.2.3 Key Categorical Features

**`id_12` (email match):**
| Value | Count |
|---|---|
| NotFound | 123,025 (85.3%) |
| Found | 21,208 (14.7%) |

**`id_15` (cookie/browser session):**
| Value | Count |
|---|---|
| Found | 67,728 |
| New | 61,612 |
| Unknown | 11,645 |

**`id_28` (browser fingerprint):**
| Value | Count |
|---|---|
| Found | 76,232 |
| New | 64,746 |

**Top `DeviceInfo` values:**
| Device | Count |
|---|---|
| Windows | 47,722 |
| iOS Device | 19,782 |
| MacOS | 12,573 |
| Trident/7.0 (IE11) | 7,440 |

> [!TIP]
> The raw `DeviceInfo` string contains browser name, version, and OS. Parsing this field to extract OS family, browser name, and browser version as separate features is highly recommended and likely to improve model performance.

#### 2.2.4 Missing Data in Identity Table

- **Total missing values:** 2,104,107
- **Columns with >50% missing:** 12

**Severely missing columns:**
| Column | Missing % |
|---|---|
| `id_24` | 96.7% |
| `id_25` | 96.4% |
| `id_07`, `id_08` | 96.4% |
| `id_21–id_23`, `id_26–id_27` | ~96.4% |
| `id_18` | 68.7% |
| `id_03`, `id_04` | 54.0% |

> [!CAUTION]
> Columns `id_21` through `id_27` are missing in ~96% of identity rows. These should be flagged with binary presence indicators and their imputed values treated with extreme caution. Consider dropping `id_24` and `id_25` entirely unless you find strong fraud signal.

---

### 2.3 `test_transaction.csv` — Test Set Transactions

**Shape:** 506,691 rows × 393 columns  
**Note:** 1 fewer column than training — no `isFraud` target column (as expected)  
**Missing values:** 73,490,163 | Columns with >50% missing: 170

The test set represents the ~6-month period immediately following the training window. The TransactionID range begins at 3,663,549 (vs. training max of ~3,577,392), confirming temporal ordering.

> [!IMPORTANT]
> The test set covers a **different time window** than training. Any time-dependent features (e.g., day of week, hour of day derived from `TransactionDT`) should be computed consistently. Be careful with aggregated statistics computed over training data — do not include test data in those computations.

---

### 2.4 `test_identity.csv` — Test Set Identity Supplement

**Shape:** 141,907 rows × 41 columns  
**Coverage:** ~28% of test transactions have identity records  
**Missing values:** 2,105,385 | Columns with >50% missing: 15

> [!NOTE]
> The column naming convention differs slightly: training identity uses underscores (`id_01`) while test identity uses hyphens (`id-01`). This is a **known Kaggle quirk** in this dataset. Ensure your preprocessing pipeline normalizes column names before merging or feature engineering.

---

### 2.5 `sample_submission.csv` — Submission Template

**Shape:** 506,691 rows × 2 columns (`TransactionID`, `isFraud`)  
**Missing values:** 0  
**Default prediction:** 0.5 (uninformed baseline)

This file defines the required output format. Predictions should be **continuous probabilities** (0.0–1.0), not binary labels. The competition metric is **AUC-ROC**.

---

## 3. Cross-Dataset Consistency

| Metric | Training | Test |
|---|---|---|
| Transaction rows | 590,540 | 506,691 |
| Transaction columns | 394 (incl. target) | 393 |
| Identity rows | 144,233 | 141,907 |
| Identity columns | 41 | 41 |
| Identity coverage | 24.4% | 28.0% |
| Time span | ~Days 1–182 | ~Days 182+ |
| Column name consistency | `id_XX` (underscore) | `id-XX` (hyphen) ⚠️ |

---

## 4. Critical Issues & Recommendations

### 4.1 Data Quality Issues

| Issue | Severity | Recommendation |
|---|---|---|
| Severe class imbalance (3.5% fraud) | 🔴 High | Use stratified sampling, class weights, or SMOTE |
| 174 features with >50% missing | 🔴 High | Missingness indicators + group-level imputation |
| `id_XX` vs `id-XX` naming mismatch | 🟠 Medium | Normalize column names in preprocessing |
| `dist2` (93.6% missing) | 🟠 Medium | Drop or use binary presence flag only |
| `anonymous.com` email domain | 🟡 Low | Engineer explicit "anonymous email" flag |

### 4.2 Feature Engineering Priorities

1. **Time features** — Extract hour-of-day, day-of-week from `TransactionDT` (fraud may be time-of-day dependent)
2. **Log-transform `TransactionAmt`** — Reduce right-skew before feeding into linear/tree models
3. **Email domain engineering** — TLD extraction, flag disposable/anonymous domains, flag domain match between P and R
4. **`DeviceInfo` parsing** — Extract OS family, browser name, browser version
5. **Missingness flags** — Binary indicator for each heavily-missing column
6. **Aggregated card features** — Count of unique emails/addresses per `card1`, mean amount per card, etc.
7. **V-feature grouping** — The 339 V-features form correlated subgroups (V1–V11, V12–V34, etc.); PCA or group-level summarization may help

### 4.3 Modelling Recommendations

- **Do not use random CV splits** — use time-based splits (chronological holdout or rolling window)
- **Primary metric:** AUC-ROC (competition standard); also track Precision-Recall AUC given imbalance
- **Recommended baseline models:** LightGBM or XGBoost with `scale_pos_weight` tuned to class ratio
- **Memory management:** The train_transaction table uses ~2.2 GB in memory; use `dtype` optimization (downcast floats to float32, ints to int32) to reduce to ~1.1 GB

---

## 5. Summary Statistics Dashboard

```
╔══════════════════════════════════════════════════════════╗
║          DATASET SUMMARY AT A GLANCE                    ║
╠══════════════════════════════════════════════════════════╣
║  Total training transactions   :   590,540              ║
║  Fraudulent transactions        :    20,663  (3.50%)    ║
║  Legitimate transactions        :   569,877  (96.50%)   ║
║  Training time span             :   ~182 days           ║
║  Avg. transaction amount        :   $135.03             ║
║  Median transaction amount      :    $68.77             ║
║  Most common card network       :   Visa (65.2%)        ║
║  Most common card type          :   Debit (74.5%)       ║
║  Most common product            :   W (74.5%)           ║
║  Top email domain               :   gmail.com           ║
║  Identity table coverage        :   24.4% (train)       ║
║  Top device type                :   Desktop (59%)       ║
║  Features with >50% missing     :   174 / 394           ║
║  Total missing cells (train_tx) :   95.6 million        ║
╚══════════════════════════════════════════════════════════╝
```

---

*Report generated from raw data exploration. No transformations have been applied to the source data. All figures are from the unprocessed `data/raw/` files.*
