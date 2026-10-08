"""
Data utilities for the IEEE-CIS Fraud Detection project.

Responsibilities:
1. Resolve project paths safely.
2. Reject missing or Git-LFS pointer files.
3. Load CSV and Parquet files.
4. Validate transaction and identity data.
5. Merge transaction + identity safely.
6. Create chronological train/validation/holdout splits.
7. Produce reproducibility reports.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import hashlib
import json
import re

import numpy as np
import pandas as pd


# ============================================================
# 1. Project paths and file helpers
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LFS_POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1"


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    """Resolve a path from the project root, not the current working directory."""
    path = Path(path).expanduser()

    if path.is_absolute():
        return path.resolve()

    base = PROJECT_ROOT if root is None else Path(root).resolve()
    return (base / path).resolve()


def require_real_file(path: str | Path) -> Path:
    """Ensure the file exists and is not just a Git LFS pointer."""
    file_path = resolve_path(path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Data file is missing:\n{file_path}\n\n"
            "Please supply the genuine IEEE-CIS dataset file."
        )

    with file_path.open("rb") as file:
        first_bytes = file.read(256)

    if first_bytes.startswith(LFS_POINTER_HEADER):
        raise ValueError(
            f"Git LFS pointer detected instead of real data:\n{file_path}\n\n"
            "The actual dataset file is not present."
        )

    return file_path


def read_table(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Parquet file safely."""
    file_path = require_real_file(path)
    extension = file_path.suffix.lower()

    if extension == ".csv":
        return pd.read_csv(file_path)

    if extension == ".parquet":
        return pd.read_parquet(file_path)

    raise ValueError(
        f"Unsupported file format: {extension}\n"
        "Expected .csv or .parquet"
    )


def calculate_file_sha256(file_path: str | Path) -> str:
    """Return SHA-256 checksum of a file, reading in chunks.

    Kept in this module so low-level file helpers live together and so
    callers (including ``__main__``) can import it from ``.data``.
    """
    sha256 = hashlib.sha256()

    with Path(file_path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# 2. Data manager
# ============================================================

class DataManager:
    """Central place for project data paths and file operations."""

    def __init__(self, root: str | Path | None = None):
        self.root = PROJECT_ROOT if root is None else Path(root).resolve()

        self.raw = self.root / "data" / "raw"
        self.processed = self.root / "data" / "processed" / "local_foundation"
        self.features = self.processed / "features"

    def ensure_directories(self) -> None:
        """Create project data directories if needed."""
        for directory in (self.raw, self.processed, self.features):
            directory.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self):
        """Load the four original IEEE-CIS tables."""
        return (
            read_table(self.raw / "train_transaction.csv"),
            read_table(self.raw / "train_identity.csv"),
            read_table(self.raw / "test_transaction.csv"),
            read_table(self.raw / "test_identity.csv"),
        )

    @staticmethod
    def _output_paths(
        directory: Path,
        kind: str,
        suffix: str = "",
    ) -> tuple[Path, Path]:
        """Build output file names without writing anything."""
        if "/" in suffix or "\\" in suffix:
            raise ValueError("suffix must only be a file-name component")

        extra = f"_{suffix}" if suffix else ""

        train_path = directory / f"train_{kind}_v2{extra}.parquet"
        test_path = directory / f"test_{kind}_v2{extra}.parquet"

        return train_path, test_path

    @staticmethod
    def _save_pair(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        train_path: Path,
        test_path: Path,
    ):
        """Save train/test parquet files without overwriting existing files."""
        if train_path.exists():
            raise FileExistsError(f"File already exists: {train_path}")

        if test_path.exists():
            raise FileExistsError(f"File already exists: {test_path}")

        train_path.parent.mkdir(parents=True, exist_ok=True)

        train_data.to_parquet(train_path, index=False)
        test_data.to_parquet(test_path, index=False)

        return str(train_path), str(test_path)

    def save_processed_data(self, train_data, test_data, suffix=""):
        paths = self._output_paths(self.processed, "data_processed", suffix)
        return self._save_pair(train_data, test_data, *paths)

    def load_processed_data(self, suffix=""):
        train_path, test_path = self._output_paths(
            self.processed, "data_processed", suffix
        )
        return read_table(train_path), read_table(test_path)

    def save_features(self, train_data, test_data, suffix=""):
        paths = self._output_paths(self.features, "features", suffix)
        return self._save_pair(train_data, test_data, *paths)

    def load_features(self, suffix=""):
        train_path, test_path = self._output_paths(
            self.features, "features", suffix
        )
        return read_table(train_path), read_table(test_path)


# ============================================================
# 3. Fingerprinting and validation helpers
# ============================================================

def dataframe_fingerprint(dataframe: pd.DataFrame) -> str:
    """Create a SHA-256 fingerprint for a dataframe."""
    hasher = hashlib.sha256()

    schema = [
        (str(column), str(dtype))
        for column, dtype in dataframe.dtypes.items()
    ]
    hasher.update(json.dumps(schema).encode())

    hashed_values = pd.util.hash_pandas_object(dataframe, index=False)
    hasher.update(hashed_values.to_numpy().tobytes())

    return hasher.hexdigest()


def _validate_columns(dataframe: pd.DataFrame, dataset_name: str) -> None:
    """Ensure column names are unique."""
    if not dataframe.columns.is_unique:
        raise ValueError(f"{dataset_name} contains duplicate column names.")


def _validate_transaction_ids(dataframe: pd.DataFrame, dataset_name: str) -> None:
    """Validate TransactionID before using it as a join key."""
    _validate_columns(dataframe, dataset_name)

    if "TransactionID" not in dataframe.columns:
        raise ValueError(f"{dataset_name} is missing TransactionID.")

    ids = dataframe["TransactionID"]

    if ids.empty:
        return

    if ids.isna().any():
        raise ValueError(f"{dataset_name} contains missing TransactionID values.")

    if ids.duplicated().any():
        raise ValueError(f"{dataset_name} contains duplicate TransactionID values.")

    if not pd.api.types.is_numeric_dtype(ids):
        raise ValueError(f"{dataset_name} TransactionID must be numeric.")

    if (ids < 0).any():
        raise ValueError(f"{dataset_name} TransactionID cannot be negative.")

    numeric_ids = ids.to_numpy(dtype=float)

    if not np.isfinite(numeric_ids).all():
        raise ValueError(
            f"{dataset_name} TransactionID contains infinite or invalid values."
        )

    if not np.equal(numeric_ids, np.floor(numeric_ids)).all():
        raise ValueError(f"{dataset_name} TransactionID must contain integers.")


def normalize_identity_columns(identity: pd.DataFrame) -> pd.DataFrame:
    """Convert test-style columns like id-30 into id_30."""
    _validate_columns(identity, "identity")

    new_columns = []
    for column in identity.columns:
        if isinstance(column, str):
            new_columns.append(re.sub(r"^id-(\d+)$", r"id_\1", column))
        else:
            new_columns.append(column)

    if len(new_columns) != len(set(new_columns)):
        raise ValueError(
            "Identity column normalization created duplicate column names."
        )

    normalized = identity.copy()
    normalized.columns = new_columns
    return normalized


def validate_transactions(
    transactions: pd.DataFrame,
    labeled: bool = True,
) -> None:
    """Validate important IEEE-CIS transaction columns."""
    _validate_transaction_ids(transactions, "transactions")

    required = ["TransactionDT", "TransactionAmt"]
    if labeled:
        required.append("isFraud")

    missing = [column for column in required if column not in transactions.columns]
    if missing:
        raise ValueError(f"Missing required transaction columns: {missing}")

    if not labeled and "isFraud" in transactions.columns:
        raise ValueError(
            "Unlabeled Kaggle test data unexpectedly contains isFraud."
        )

    for column in ("TransactionDT", "TransactionAmt"):
        values = transactions[column]

        if not pd.api.types.is_numeric_dtype(values):
            raise ValueError(f"{column} must be numeric.")

        if values.isna().any():
            raise ValueError(f"{column} contains missing values.")

        numeric_values = values.to_numpy(dtype=float)

        if not np.isfinite(numeric_values).all():
            raise ValueError(f"{column} contains infinite values.")

        if (values < 0).any():
            raise ValueError(f"{column} cannot contain negative values.")

    if labeled:
        labels = transactions["isFraud"]

        if labels.isna().any():
            raise ValueError("isFraud contains missing values.")

        if not labels.isin([0, 1]).all():
            raise ValueError("isFraud must contain only 0 and 1.")


def missingness_report(dataframe: pd.DataFrame) -> dict:
    """Measure missing values without modifying the dataframe."""
    report = {}

    for column in dataframe.columns:
        missing_count = int(dataframe[column].isna().sum())
        missing_fraction = (
            float(dataframe[column].isna().mean())
            if len(dataframe) > 0
            else 0.0
        )

        report[str(column)] = {
            "missing_count": missing_count,
            "missing_fraction": missing_fraction,
        }

    return report


# ============================================================
# 4. Merge transaction + identity
# ============================================================

def merge_transaction_identity(
    transactions: pd.DataFrame,
    identity: pd.DataFrame,
    labeled: bool = True,
):
    """Safely perform a one-to-one left join."""
    validate_transactions(transactions, labeled=labeled)
    identity = normalize_identity_columns(identity)
    _validate_transaction_ids(identity, "identity")

    overlap = set(transactions.columns) & set(identity.columns)
    overlap.discard("TransactionID")

    if overlap:
        raise ValueError(
            f"Transaction and identity tables have overlapping columns: {sorted(overlap)}"
        )

    # Pick an indicator column name that cannot collide with real columns.
    indicator = "__identity_match__"
    while indicator in transactions.columns or indicator in identity.columns:
        indicator = "_" + indicator

    merged = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=indicator,
    )

    if len(merged) != len(transactions):
        raise RuntimeError("Merge changed the number of transaction rows.")

    original_ids = transactions["TransactionID"].reset_index(drop=True)
    merged_ids = merged["TransactionID"].reset_index(drop=True)

    if not merged_ids.equals(original_ids):
        raise RuntimeError("Merge changed transaction order.")

    matched = int(merged[indicator].eq("both").sum())
    unmatched = len(transactions) - matched
    orphan = int(
        (~identity["TransactionID"].isin(transactions["TransactionID"])).sum()
    )

    coverage = matched / len(transactions) if len(transactions) > 0 else 0.0

    merged = merged.drop(columns=[indicator])

    report = {
        "schema_version": 2,
        "labeled": labeled,
        "transaction_rows": len(transactions),
        "identity_rows": len(identity),
        "merged_rows": len(merged),
        "identity_matched_rows": matched,
        "identity_unmatched_transaction_rows": unmatched,
        "identity_orphan_rows": orphan,
        "identity_match_coverage": coverage,
        "transaction_fingerprint": dataframe_fingerprint(transactions),
        "identity_fingerprint": dataframe_fingerprint(identity),
        "merged_fingerprint": dataframe_fingerprint(merged),
        "original_missingness": missingness_report(merged),
    }

    return merged, report


# ============================================================
# 5. Chronological split
# ============================================================

@dataclass(frozen=True)
class Partitions:
    """Container returned by chronological_split()."""

    train: pd.DataFrame
    validation: pd.DataFrame
    holdout: pd.DataFrame
    report: dict


def _partition_report(dataframe: pd.DataFrame) -> dict:
    """Create basic metadata for one dataset partition."""
    fraud_count = int(dataframe["isFraud"].eq(1).sum())
    non_fraud_count = int(dataframe["isFraud"].eq(0).sum())

    return {
        "rows": len(dataframe),
        "elapsed_seconds_min": float(dataframe["TransactionDT"].min()),
        "elapsed_seconds_max": float(dataframe["TransactionDT"].max()),
        "class_counts": {
            "0": non_fraud_count,
            "1": fraud_count,
        },
        "fingerprint": dataframe_fingerprint(dataframe),
    }


def chronological_split(
    labeled_pool: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    train_end: float | None = None,
    validation_end: float | None = None,
):
    """
    Split labeled transactions chronologically.

    Default:
        ~60% train
        ~20% validation
        ~20% holdout

    Rows sharing the same TransactionDT stay in the same partition.
    """
    validate_transactions(labeled_pool, labeled=True)

    ordered = (
        labeled_pool
        .sort_values("TransactionDT", kind="stable")
        .reset_index(drop=True)
    )

    manual = train_end is not None or validation_end is not None

    if manual:
        if train_end is None or validation_end is None:
            raise ValueError("Both train_end and validation_end must be supplied.")

        if not np.isfinite([train_end, validation_end]).all():
            raise ValueError("Chronological boundaries must be finite.")

        if train_end < 0:
            raise ValueError("train_end cannot be negative.")

        if train_end >= validation_end:
            raise ValueError("train_end must be earlier than validation_end.")

        train_end = float(train_end)
        validation_end = float(validation_end)
        split_mode = "explicit_elapsed_seconds"

    else:
        if train_fraction <= 0:
            raise ValueError("train_fraction must be greater than 0.")

        if validation_fraction <= 0:
            raise ValueError("validation_fraction must be greater than 0.")

        if train_fraction + validation_fraction >= 1:
            raise ValueError(
                "Train + validation fractions must leave room for holdout."
            )

        timestamp_sizes = ordered.groupby("TransactionDT", sort=True).size()

        if len(timestamp_sizes) < 3:
            raise ValueError("At least three different timestamps are required.")

        cumulative_rows = timestamp_sizes.cumsum().to_numpy()
        total_rows = len(ordered)

        desired_train_rows = total_rows * train_fraction
        desired_validation_end_rows = total_rows * (
            train_fraction + validation_fraction
        )

        train_candidates = cumulative_rows[:-2]
        train_group_index = int(
            np.argmin(np.abs(train_candidates - desired_train_rows))
        )

        validation_candidate_indexes = np.arange(
            train_group_index + 1,
            len(timestamp_sizes) - 1,
        )

        validation_distances = np.abs(
            cumulative_rows[validation_candidate_indexes]
            - desired_validation_end_rows
        )

        validation_group_index = int(
            validation_candidate_indexes[np.argmin(validation_distances)]
        )

        train_end = float(timestamp_sizes.index[train_group_index])
        validation_end = float(timestamp_sizes.index[validation_group_index])
        split_mode = "nearest_timestamp_group_fractions"

    train = ordered[ordered["TransactionDT"] <= train_end].copy()
    validation = ordered[
        (ordered["TransactionDT"] > train_end)
        & (ordered["TransactionDT"] <= validation_end)
    ].copy()
    holdout = ordered[ordered["TransactionDT"] > validation_end].copy()

    partitions = {
        "train": train,
        "validation": validation,
        "holdout": holdout,
    }

    for name, dataframe in partitions.items():
        if dataframe.empty:
            raise ValueError(
                f"{name} partition is empty. Increase the source sample, "
                "reduce the boundary fractions, or supply explicit "
                "train_end/validation_end values."
            )

        if set(dataframe["isFraud"].unique()) != {0, 1}:
            raise ValueError(
                f"{name} must contain both fraud and non-fraud rows. "
                "The source period or row cap is too small for a "
                "chronological split with both classes in every partition."
            )

    if train["TransactionDT"].max() >= validation["TransactionDT"].min():
        raise RuntimeError("Train and validation time ranges overlap.")

    if validation["TransactionDT"].max() >= holdout["TransactionDT"].min():
        raise RuntimeError("Validation and holdout time ranges overlap.")

    train_ids = set(train["TransactionID"])
    validation_ids = set(validation["TransactionID"])
    holdout_ids = set(holdout["TransactionID"])

    if train_ids & validation_ids:
        raise RuntimeError("Train and validation IDs overlap.")

    if train_ids & holdout_ids:
        raise RuntimeError("Train and holdout IDs overlap.")

    if validation_ids & holdout_ids:
        raise RuntimeError("Validation and holdout IDs overlap.")

    total_rows = len(ordered)

    report = {
        "schema_version": 2,
        "split_mode": split_mode,
        "time_meaning": "relative_elapsed_seconds",
        "train_fraction_requested": train_fraction,
        "validation_fraction_requested": validation_fraction,
        "train_end": train_end,
        "validation_end": validation_end,
        "boundary_rules": (
            "train <= train_end; "
            "train_end < validation <= validation_end; "
            "holdout > validation_end"
        ),
        "labeled_pool_rows": len(labeled_pool),
        "labeled_pool_fingerprint": dataframe_fingerprint(labeled_pool),
        "partitions": {
            "train": _partition_report(train),
            "validation": _partition_report(validation),
            "holdout": _partition_report(holdout),
        },
        "actual_row_fractions": {
            "train": len(train) / total_rows,
            "validation": len(validation) / total_rows,
            "holdout": len(holdout) / total_rows,
        },
        "ids_disjoint": True,
        "boundary_timestamps_disjoint": True,
    }

    return Partitions(
        train=train,
        validation=validation,
        holdout=holdout,
        report=report,
    )