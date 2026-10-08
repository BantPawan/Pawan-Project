"""Small reporting and compatibility helpers shared by the fraud workflow.

Evaluation reports combine ranking metrics, threshold-dependent errors and
alert volume. The remaining helpers support notebook exploration and preserve
the earlier project's logging, timing and memory-inspection interfaces.
"""

import hashlib
import json
import logging
import time
from functools import wraps

import numpy as np
import pandas as pd


def column_view(frame, columns):
    """Select columns without duplicating full numeric blocks.

    Internal consumers must treat this view as read-only. Transformers build
    their own outputs before modifying values.
    """
    return pd.DataFrame(
        {name: frame[name] for name in columns}, index=frame.index, copy=False
    )


def numeric_feature_view(frame, columns=None):
    """Convert numeric columns and clean infinities one column at a time.

    Avoid pandas' whole-frame replace/consolidation temporaries. Already-clean
    float64 columns are shared; changed columns get independent arrays.
    """
    columns = frame.columns if columns is None else columns
    values = {}
    for name in columns:
        array = frame[name].to_numpy(dtype=np.float64, na_value=np.nan)
        infinite = np.isinf(array)
        if infinite.any():
            array = array.copy()
            array[infinite] = np.nan
        values[name] = array
    return pd.DataFrame(values, index=frame.index, copy=False)


def setup_logging(log_file="pipeline.log"):
    """Send informational workflow messages to both a file and the console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def timer(func):
    """Decorate a function to print its elapsed execution time after it returns."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result

    return wrapper


def calculate_metrics(y_true, y_pred, y_prob):
    """Describe probability ranking and the errors from a chosen decision threshold.

    ``y_pred`` contains the already thresholded labels; this function does not
    choose a threshold. ROC-AUC and average precision are omitted for a
    single-class slice. Confusion counts and alert volume remain useful for
    those slices, and zero predicted positives produces zero precision.
    """
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        brier_score_loss,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    if not (len(y_true) == len(y_pred) == len(y_prob)) or not len(y_true):
        raise ValueError("Evaluation arrays must have equal, nonzero lengths")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Evaluation labels must be binary")
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    both_classes = len(np.unique(y_true)) == 2
    actual_negative_count = false_positive + true_negative
    alert_count = false_positive + true_positive
    row_count = len(y_true)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else None,
        "average_precision": (
            float(average_precision_score(y_true, y_prob)) if both_classes else None
        ),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
        "false_positive_rate": (
            float(false_positive / actual_negative_count)
            if actual_negative_count
            else None
        ),
        "predicted_positive": int(alert_count),
        "rows": int(row_count),
        "false_positives_per_1000_transactions": float(
            1000 * false_positive / row_count
        ),
        "alerts_per_1000_transactions": float(1000 * alert_count / row_count),
        "fraud_prevalence": float(np.mean(y_true)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": [
            [int(true_negative), int(false_positive)],
            [int(false_negative), int(true_positive)],
        ],
    }

    return metrics


def plot_feature_importance(feature_importance, feature_names, top_n=20):
    """Display the largest feature-importance values alongside their column names."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    count = min(top_n, len(feature_importance))
    indices = np.argsort(feature_importance)[-count:]

    plt.barh(range(count), feature_importance[indices])
    plt.yticks(range(count), [feature_names[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.show()


def memory_usage(df):
    """Return the DataFrame's estimated memory use in MiB, including object values."""
    return df.memory_usage(deep=True).sum() / 1024**2  # MB


def reduce_memory_usage(df):
    """Downcast integer columns in place when their observed range fits safely.

    Floating-point columns keep their precision: a small observed range cannot
    establish whether downcasting will preserve amounts or elapsed timestamps.
    The returned frame is the same object supplied by the caller.
    """
    starting_memory = memory_usage(df)

    for column in df.columns:
        column_type = df[column].dtype

        if column_type != object:
            minimum = df[column].min()
            maximum = df[column].max()

            if str(column_type)[:3] == "int":
                if minimum > np.iinfo(np.int8).min and maximum < np.iinfo(np.int8).max:
                    df[column] = df[column].astype(np.int8)
                elif (
                    minimum > np.iinfo(np.int16).min
                    and maximum < np.iinfo(np.int16).max
                ):
                    df[column] = df[column].astype(np.int16)
                elif (
                    minimum > np.iinfo(np.int32).min
                    and maximum < np.iinfo(np.int32).max
                ):
                    df[column] = df[column].astype(np.int32)
                elif (
                    minimum > np.iinfo(np.int64).min
                    and maximum < np.iinfo(np.int64).max
                ):
                    df[column] = df[column].astype(np.int64)
            else:
                # Range alone cannot establish safe precision for amounts or timestamps.
                continue

    ending_memory = memory_usage(df)
    reduction = (starting_memory - ending_memory) / starting_memory * 100
    print(
        f"Memory reduced from {starting_memory:.2f} MB to {ending_memory:.2f} MB "
        f"({reduction:.1f}% reduction)"
    )

    return df


def validate_data_splits(X_train, X_val, y_train, y_val):
    """Check feature/label lengths, column order and observable split boundaries.

    When raw IDs and elapsed timestamps are present, also reject shared IDs and
    overlapping time windows. This compatibility helper does not replace the
    full transaction validation performed before chronological splitting.
    """
    if len(X_train) != len(y_train) or len(X_val) != len(y_val):
        raise ValueError("Features and labels length mismatch")
    if list(X_train.columns) != list(X_val.columns):
        raise ValueError("Train and validation feature order mismatch")
    if "TransactionID" in X_train and "TransactionID" in X_val:
        if set(X_train.TransactionID) & set(X_val.TransactionID):
            raise ValueError("Overlapping transaction IDs")
    if "TransactionDT" in X_train and "TransactionDT" in X_val:
        if X_train.TransactionDT.max() >= X_val.TransactionDT.min():
            raise ValueError("Overlapping or unordered boundary timestamps")

    print("✓ Data splits validated successfully")
    print(f"  Train: {X_train.shape} features, {y_train.shape} labels")
    print(f"  Validation: {X_val.shape} features, {y_val.shape} labels")


def frame_fingerprint(frame):
    """Hash ordered values and schema while excluding the incidental DataFrame index."""
    digest = hashlib.sha256()
    schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    row_hashes = pd.util.hash_pandas_object(frame, index=False).values
    digest.update(json.dumps(schema).encode())
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()
