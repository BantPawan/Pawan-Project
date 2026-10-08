"""
Prepare validated IEEE-CIS fraud data and create chronological partitions.

This module does NOT fit preprocessing rules such as imputation, scaling,
encoding, or feature selection. Those must be fitted on training data only.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .data import (
    DataManager,
    PROJECT_ROOT,
    Partitions,
    calculate_file_sha256,
    chronological_split,
    merge_transaction_identity,
    validate_transactions,
)


# Scopes that identify genuine source data. Generated/fixture scopes are
# excluded so a synthetic dataset cannot be validated as real evidence.
ALLOWED_MANIFEST_SCOPES = ("real_sample", "real_full")

# Frames written by ``save_data`` and the report names written beside them.
PREPARED_FRAMES = ("labeled_pool", "kaggle_test", "train", "validation", "holdout")
PREPARED_REPORTS = ("data_quality", "split")


def validate_raw_manifest(raw_dir: str | Path, filenames=None):
    """Validate data against ``data_manifest.json`` if it exists.

    Returns the parsed manifest, or an empty dict when no manifest is present.
    A present manifest must declare a genuine scope in
    ``ALLOWED_MANIFEST_SCOPES`` and must contain a checksum for every listed
    file. The checksum of each on-disk file is then verified against it.
    """
    raw_directory = Path(raw_dir)
    manifest_path = raw_directory / "data_manifest.json"

    if not manifest_path.exists():
        return {}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    scope = manifest.get("data_scope")
    if scope not in ALLOWED_MANIFEST_SCOPES:
        raise ValueError(
            "Raw directory manifest must declare a genuine data scope in "
            f"{ALLOWED_MANIFEST_SCOPES}, got {scope!r}."
        )

    if filenames is None:
        filenames = (
            "train_transaction.csv",
            "train_identity.csv",
            "test_transaction.csv",
            "test_identity.csv",
        )
    elif isinstance(filenames, str):
        filenames = (filenames,)

    for filename in filenames:
        file_path = raw_directory / filename

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Manifest expects this file, but it is missing:\n{file_path}"
            )

        try:
            expected_sha256 = manifest["files"][filename]["local_sha256"]
        except KeyError as error:
            raise ValueError(
                f"Manifest does not contain a checksum for {filename}."
            ) from error

        actual_sha256 = calculate_file_sha256(file_path)

        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Downloaded sample differs from its source manifest:\n{filename}"
            )

    return manifest


class DataPreparation:
    """Prepare trustworthy data before model preprocessing."""

    def __init__(
        self,
        root: str | Path | None = None,
        raw_dir: str | Path | None = None,
    ):
        self.root = PROJECT_ROOT if root is None else Path(root).resolve()
        self.manager = DataManager(self.root)

        # ``DataManager`` reads raw files from ``self.raw`` and writes
        # prepared data under ``self.processed``. Override those, not the
        # non-existent DATA_*_PATH names, so ``load_raw_data`` honors the
        # caller's directory.
        if raw_dir is not None:
            self.manager.raw = Path(raw_dir).resolve()

        # Only create the destination directory. The raw source directory is
        # never created: a missing raw directory must stay a clear error.
        self.manager.processed.mkdir(parents=True, exist_ok=True)

        self.DATA_RAW_PATH = self.manager.raw
        self.DATA_PROCESSED_PATH = self.manager.processed

        self.data_scope = "unrecorded"
        self.quality_report = {}
        self.split_report = {}
        self.output_paths = {}
        self.source_manifest = {}

    def load_data(self):
        """Load the four original IEEE-CIS raw tables.

        Derives the recorded data scope from an optional source manifest.
        Without a manifest, the scope defaults to ``real_full`` because
        genuine raw files are being read.
        """
        self.source_manifest = validate_raw_manifest(self.DATA_RAW_PATH)

        if self.source_manifest:
            self.data_scope = self.source_manifest.get("data_scope", "real_full")
        else:
            self.data_scope = "real_full"

        return self.manager.load_raw_data()

    def merge_datasets(
        self,
        train_transaction: pd.DataFrame,
        train_identity: pd.DataFrame,
        test_transaction: pd.DataFrame,
        test_identity: pd.DataFrame,
    ):
        """Create labeled_pool and unlabeled kaggle_test."""
        labeled_pool, labeled_report = merge_transaction_identity(
            transactions=train_transaction,
            identity=train_identity,
            labeled=True,
        )

        kaggle_test, kaggle_test_report = merge_transaction_identity(
            transactions=test_transaction,
            identity=test_identity,
            labeled=False,
        )

        if set(labeled_pool["TransactionID"]) & set(kaggle_test["TransactionID"]):
            raise ValueError(
                "Labeled pool and Kaggle test contain overlapping TransactionID values."
            )

        self.quality_report = {
            "labeled_pool": labeled_report,
            "kaggle_test": kaggle_test_report,
        }

        if self.source_manifest:
            self.quality_report["source_manifest"] = self.source_manifest

        return labeled_pool, kaggle_test

    @staticmethod
    def analyze_missingness(
        dataframe: pd.DataFrame,
        dataset_name: str = "dataset",
    ):
        """Analyze missing-value percentages. Does not modify data."""
        total_rows = len(dataframe)

        print(f"Missingness analysis: {dataset_name}")
        print(f"Rows: {total_rows:,}")

        missing_fraction = dataframe.isna().mean()
        return (missing_fraction * 100).sort_values(ascending=False)

    @staticmethod
    def analyze_target(train_data: pd.DataFrame):
        """Return class percentages for isFraud (0/1)."""
        validate_transactions(train_data, labeled=True)

        labels = train_data["isFraud"]
        class_counts = labels.value_counts().reindex([0, 1], fill_value=0)
        total_rows = len(train_data)

        if total_rows == 0:
            return class_counts.astype(float)

        return class_counts / total_rows * 100

    def save_data(
        self,
        labeled_pool: pd.DataFrame,
        kaggle_test: pd.DataFrame,
        partitions: Partitions,
        data_scope: str | None = None,
    ):
        """Save prepared data in a unique run directory.

        Writes each frame as a Parquet file, each aggregate report as JSON,
        and a ``prepared_manifest.json`` that records the data scope, file
        checksums and row/column counts so ``load_prepared_data`` can
        reconstruct and verify the same run.
        """
        scope = data_scope or self.data_scope

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        unique_id = uuid4().hex[:8]
        run_name = f"v2_{timestamp}_{unique_id}"

        run_directory = self.DATA_PROCESSED_PATH / run_name
        run_directory.mkdir(parents=True, exist_ok=False)

        frames_to_save = {
            "labeled_pool": labeled_pool,
            "kaggle_test": kaggle_test,
            "train": partitions.train,
            "validation": partitions.validation,
            "holdout": partitions.holdout,
        }

        file_records: dict[str, dict] = {}
        for name, dataframe in frames_to_save.items():
            output_path = run_directory / f"{name}_v2.parquet"
            dataframe.to_parquet(output_path, index=False)
            self.output_paths[name] = str(output_path)
            file_records[output_path.name] = {
                "sha256": calculate_file_sha256(output_path),
                "rows": int(len(dataframe)),
                "columns": int(dataframe.shape[1]),
            }

        reports_to_save = {
            "data_quality": self.quality_report,
            "split": partitions.report,
        }

        for name, report in reports_to_save.items():
            report_path = run_directory / f"{name}_report_v2.json"
            report_path.write_text(
                json.dumps(report, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            self.output_paths[name] = str(report_path)

        prepared_manifest = {
            "schema_version": 2,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "data_scope": scope,
            "run_name": run_name,
            "files": file_records,
            "source_manifest": self.source_manifest or None,
        }
        manifest_path = run_directory / "prepared_manifest.json"
        manifest_path.write_text(
            json.dumps(prepared_manifest, indent=2),
            encoding="utf-8",
        )
        self.output_paths["prepared_manifest"] = str(manifest_path)

        return self.output_paths.copy()

    def run(self, data_scope: str | None = None, **split_options):
        """Run the complete data-preparation workflow.

        ``data_scope`` optionally overrides the scope recorded in the
        prepared manifest (for example, when the caller applies ``--max-rows``
        and the working frame becomes a bounded sample).
        """
        (
            train_transaction,
            train_identity,
            test_transaction,
            test_identity,
        ) = self.load_data()

        labeled_pool, kaggle_test = self.merge_datasets(
            train_transaction,
            train_identity,
            test_transaction,
            test_identity,
        )

        partitions = chronological_split(labeled_pool, **split_options)
        self.split_report = partitions.report

        self.save_data(
            labeled_pool=labeled_pool,
            kaggle_test=kaggle_test,
            partitions=partitions,
            data_scope=data_scope,
        )

        return partitions, kaggle_test


def load_prepared_data(prepared_folder: str | Path) -> dict:
    """Load a prepared run directory written by ``DataPreparation.save_data``.

    Returns a dict with the keys ``manifest``, ``partitions``,
    ``labeled_pool``, ``kaggle_test`` and ``quality_report``. The manifest
    carries ``data_scope`` and per-file ``sha256`` records; ``partitions``
    is a ``Partitions`` instance whose frames carry ``attrs['data_scope']``.
    """
    folder = Path(prepared_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Prepared folder not found: {folder}")

    manifest_path = folder / "prepared_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Prepared folder is missing prepared_manifest.json: {folder}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("schema_version") != 2:
        raise ValueError(
            "Unsupported prepared_manifest schema_version: "
            f"{manifest.get('schema_version')!r}"
        )

    scope = manifest.get("data_scope", "unrecorded")

    def _read_frame(name: str) -> pd.DataFrame:
        path = folder / f"{name}_v2.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Prepared frame is missing: {path}")

        try:
            record = manifest["files"][path.name]
            expected_sha256 = record["sha256"]
            expected_rows = int(record["rows"])
            expected_columns = int(record["columns"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Prepared manifest has no complete integrity record for {path.name}."
            ) from error

        actual_sha256 = calculate_file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Prepared file checksum differs from its manifest: {path.name}"
            )

        frame = pd.read_parquet(path)
        actual_shape = (int(len(frame)), int(frame.shape[1]))
        expected_shape = (expected_rows, expected_columns)
        if actual_shape != expected_shape:
            raise ValueError(
                "Prepared file shape differs from its manifest: "
                f"{path.name}; expected {expected_shape}, got {actual_shape}"
            )

        return frame

    labeled_pool = _read_frame("labeled_pool")
    kaggle_test = _read_frame("kaggle_test")
    train = _read_frame("train")
    validation = _read_frame("validation")
    holdout = _read_frame("holdout")

    split_report_path = folder / "split_report_v2.json"
    if split_report_path.is_file():
        split_report = json.loads(split_report_path.read_text(encoding="utf-8"))
    else:
        split_report = {}

    quality_path = folder / "data_quality_report_v2.json"
    if quality_path.is_file():
        quality_report = json.loads(quality_path.read_text(encoding="utf-8"))
    else:
        quality_report = {}

    partitions = Partitions(
        train=train,
        validation=validation,
        holdout=holdout,
        report=split_report,
    )

    for frame in (train, validation, holdout):
        frame.attrs["data_scope"] = scope

    return {
        "manifest": manifest,
        "partitions": partitions,
        "labeled_pool": labeled_pool,
        "kaggle_test": kaggle_test,
        "quality_report": quality_report,
    }


if __name__ == "__main__":
    DataPreparation().run()