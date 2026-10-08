# Network Intrusion Detection with Machine Learning

I built this project to explore how machine learning can help identify suspicious network traffic. Using the NSL-KDD dataset, I worked through data exploration, feature engineering, feature selection and model comparison to classify connections as normal traffic or an attack.

The project connects my interests in cybersecurity and AI/ML: understanding what network activity tells us, preparing useful features and evaluating how well a model catches attacks without creating too many false alerts.

## About the dataset

NSL-KDD is a benchmark dataset for network intrusion detection, developed to address problems such as repeated records in the older KDD'99 dataset. Each row describes a network connection. The raw files have **41 features**, followed by an attack label and a difficulty level. [Dataset background](https://www.unb.ca/cic/datasets/nsl.html)

The features include connection duration, protocol, service, bytes sent and received, failed login attempts and patterns in connections to the same host or service. These give the models information about both individual connections and surrounding traffic.

The files included in this project are:

| File | Records | Purpose |
| --- | ---: | --- |
| `KDDTrain+.txt` | 125,973 | Full training set |
| `KDDTest+.txt` | 22,544 | Separate test set |
| `KDDTrain+_20Percent.txt` | 25,192 | Smaller training subset |

The labels are used in two ways: **binary classification** separates normal traffic from attacks, while **multiclass classification** groups connections into normal traffic and four attack categories:

| Category | What it describes |
| --- | --- |
| Denial of Service (DoS) | Attempts to overwhelm a service or make it unavailable |
| Probe | Scanning or reconnaissance to discover hosts, services or weaknesses |
| Remote to Local (R2L) | Attempts to gain local access from a remote connection |
| User to Root (U2R) | Attempts to gain higher privileges from an existing user account |

The test set also contains attack types absent from the training set, making generalization an important part of the evaluation.

## What I worked on

- **Exploratory analysis:** examined class distributions, protocols, services, numerical features and relationships with attack labels.
- **Feature engineering:** created features such as byte ratios, connection intensity, combined error rates and privilege-related indicators.
- **Feature selection:** explored mutual information, model-based importance, permutation importance and SHAP to understand which inputs contribute to detection.
- **Model comparison:** experimented with logistic regression, Naive Bayes, Random Forest, Gradient Boosting, XGBoost and LightGBM, along with stacking and voting ensembles.
- **Imbalanced learning and tuning:** explored class weighting, SMOTE-based sampling, cross-validation, hyperparameter searches and decision thresholds.

## Evaluating the models

The notebooks contain classification reports, confusion matrices and model comparison tables. I use precision, recall and F1 alongside accuracy because the errors have different consequences in a security setting.

Recall shows how many attacks are detected. Precision helps explain how many alerts correspond to actual attacks. Looking at both makes it easier to understand the trade-off between missed attacks and the workload created by false alerts. Macro F1 also gives each class equal weight, which helps expose weak performance on less common attack categories.

The saved model artifact is in [`models/final_pipeline.pkl`](models/final_pipeline.pkl). It contains an XGBoost pipeline with preprocessing and SMOTE. Results from individual experiments are kept alongside their code in the notebooks; they should be read with the relevant model, split and threshold in mind.

## Finding your way around

| Location | Contents |
| --- | --- |
| `data/raw/` | Original NSL-KDD training and test files |
| `data/processed/` | Prepared datasets and selected-feature exports |
| `eda/` | Dataset notes and exploratory analysis |
| `eda/figures/` | Saved charts from the analysis |
| `notebooks/` | Preparation, feature selection and modeling experiments |
| `models/` | Saved model artifact |
| `src/Network_intrution/` | Utilities for saving charts |

Start with [`eda/dataset_overview.ipynb`](eda/dataset_overview.ipynb) for the feature descriptions and [`eda/nslkdd_eda.ipynb`](eda/nslkdd_eda.ipynb) for the analysis. The main work is documented in:

1. [`00_data_preparation.ipynb`](notebooks/00_data_preparation.ipynb)
2. [`01_feature_selection.ipynb`](notebooks/01_feature_selection.ipynb)
3. [`03_model_selection.ipynb`](notebooks/03_model_selection.ipynb)
4. [`experiment.ipynb`](notebooks/experiment.ipynb)

## Running locally

The project was developed with Python 3.11. For Windows PowerShell, run the following from a folder where you want to clone the repository:

```powershell
git clone https://github.com/BantPawan/Pawan-Project.git
cd "Pawan-Project\network_intrution_detection"
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\jupyter.exe notebook
```

The dependencies cover the different experiments in the repository. The main tools are Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, imbalanced-learn, SHAP, Matplotlib, Seaborn and Jupyter.

These are exploratory notebooks, so review the data-loading cells before rerunning them. The EDA notebook has a machine-specific utility import path that needs to point to your local `src/Network_intrution/` folder. The older `02_modeling_baseline.ipynb` expects `cleaned_feature_selected.parquet`, which is not included in the current exports. The processed CSV files are included for reviewing the later experiments.

## Scope and next steps

This is an offline benchmark project. NSL-KDD is an older dataset, so its results do not establish how the models would perform on current network traffic. A useful next step would be evaluation on a newer dataset and a separate validation workflow for choosing features and thresholds before the final test.

## Dataset credit

Dataset: [NSL-KDD, Canadian Institute for Cybersecurity, University of New Brunswick](https://www.unb.ca/cic/datasets/nsl.html).

M. Tavallaee, E. Bagheri, W. Lu and A. A. Ghorbani, [*A Detailed Analysis of the KDD CUP 99 Data Set*](https://ieeexplore.ieee.org/document/5356528), IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA), 2009.
