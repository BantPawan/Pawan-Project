"""MLflow tracking and registry integration for the local fraud workflow.

Local JSON, CSV and joblib artifacts remain the canonical record of every
experiment. MLflow adds three things on top of them:

1. A cross-run index so comparisons, tuning and final assessments are
   searchable without walking the filesystem.
2. A registry so a deployment can resolve the promoted model by name and
   stage instead of by an opaque local path.
3. A model signature that pins the raw-input contract a served model
   expects.

MLflow does not replace the local fingerprint lineage. The
``partition_fingerprints`` and ``source_file_sha256`` values are written
as run tags so a promoted model can be re-verified against the data it
was fit on. A model should only be promoted to ``Production`` after those
tags match the current training data.

Tracking is optional: the workflow runs locally without MLflow installed.
Callers opt in with ``--track`` on ``compare``, ``train`` and ``finalize``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PROJECT_ROOT


DEFAULT_EXPERIMENT = "ieee_cis_fraud_detection"
DEFAULT_REGISTERED_MODEL = "ieee_cis_fraud_detector"
DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"

# MLflow enforces a length limit on string parameters. Long JSON values are
# truncated here so logging never fails on a wide feature recipe.
MAX_PARAM_LENGTH = 500


def _require_mlflow():
    try:
        import mlflow
    except ImportError as error:
        raise ImportError(
            "MLflow is not installed. Install requirements-mlops.txt or run "
            "the workflow without --track."
        ) from error
    return mlflow


def configure_tracking(tracking_uri=None, experiment=None):
    """Bind this process to a tracking URI and experiment.

    A SQLite file inside the project root is the default backend, so a
    fresh checkout works without a running server. Returns the resolved
    tracking URI so callers can report it.
    """
    mlflow = _require_mlflow()
    uri = tracking_uri or DEFAULT_TRACKING_URI
    name = experiment or DEFAULT_EXPERIMENT

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(name)
    return uri


def _to_param_text(value):
    """Coerce any report value into an MLflow-safe parameter string."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        text = json.dumps(list(value), default=str)
    elif isinstance(value, dict):
        text = json.dumps(value, default=str, sort_keys=True)
    else:
        text = str(value)

    if len(text) > MAX_PARAM_LENGTH:
        text = text[: MAX_PARAM_LENGTH - 3] + "..."
    return text


def _log_scalar_metrics(mlflow, prefix, mapping):
    """Log every finite numeric entry of a metrics mapping under ``prefix``."""
    if not isinstance(mapping, dict):
        return
    for key, value in mapping.items():
        if key == "confusion_matrix" or value is None:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            number = float(value)
            if np.isfinite(number):
                mlflow.log_metric(f"{prefix}_{key}", number)


def _log_common_lineage(mlflow, metadata, extra_tags=None):
    """Record the lineage tags any promoted model must match later."""
    tags = {
        "partition_fingerprints": json.dumps(
            metadata.get("partition_fingerprints", {}), sort_keys=True
        ),
        "source_file_sha256": json.dumps(
            metadata.get("source_file_sha256", {}), sort_keys=True
        ),
        "selection_criterion": metadata.get(
            "selection_criterion", "explicit single candidate"
        ),
    }
    if extra_tags:
        tags.update({str(k): str(v) for k, v in extra_tags.items()})
    mlflow.set_tags(tags)


def log_comparison(
    comparison,
    folder,
    *,
    tracking_uri=None,
    experiment=None,
):
    """Log one comparison run: params, candidate metrics and report files.

    ``folder`` is the directory written by ``save_comparison``. The
    ``comparison.json`` and ``comparison.csv`` files are attached as run
    artifacts. Per-candidate validation metrics are logged so the MLflow UI
    can sort candidates without opening the JSON.
    """
    mlflow = _require_mlflow()
    configure_tracking(tracking_uri, experiment)

    folder = Path(folder)
    metadata = comparison.metadata
    winner = comparison.winner

    params = {
        "stage": "comparison",
        "comparison_id": metadata["comparison_id"],
        "data_scope": metadata.get("data_scope", "unrecorded"),
        "selection_criterion": metadata["selection_criterion"],
        "tie_policy": metadata.get("tie_policy", ""),
        "seed": metadata["seed"],
        "n_splits": metadata["n_splits"],
        "winner_name": comparison.winner_name,
        "winner_artifact_id": winner.metadata["artifact_id"],
        "holdout_evaluated": metadata.get("holdout_evaluated", False),
        "requested_configs": json.dumps(
            metadata.get("requested_configs", {}), default=str, sort_keys=True
        ),
    }

    with mlflow.start_run(
        run_name=f"comparison_{metadata['comparison_id'][:8]}"
    ) as run:
        mlflow.log_params({k: _to_param_text(v) for k, v in params.items()})
        _log_common_lineage(
            mlflow,
            metadata,
            extra_tags={
                "workflow_type": metadata.get("workflow_type", "comparison"),
            },
        )

        # Per-candidate validation metrics for the MLflow UI.
        for _, row in comparison.table.iterrows():
            candidate = str(row["candidate"])
            for metric_name in (
                "validation_average_precision",
                "validation_roc_auc",
                "validation_f1",
                "validation_precision",
                "validation_recall",
            ):
                if metric_name in row and pd.notna(row[metric_name]):
                    mlflow.log_metric(
                        f"{metric_name}/{candidate}", float(row[metric_name])
                    )

        top = comparison.table.iloc[0]
        for name, column in (
            ("winner_average_precision", "validation_average_precision"),
            ("winner_roc_auc", "validation_roc_auc"),
        ):
            if column in top and pd.notna(top[column]):
                mlflow.log_metric(name, float(top[column]))

        for filename in ("comparison.json", "comparison.csv"):
            path = folder / filename
            if path.is_file():
                mlflow.log_artifact(str(path))

        return run.info.run_id


def log_result(
    result,
    partitions,
    saved_pipeline_path,
    folder,
    *,
    comparison=None,
    data_scope=None,
    tracking_uri=None,
    experiment=None,
    register_name=None,
    register_stage=None,
):
    """Log one fitted candidate: metrics, lineage, model and reports.

    The model is logged with a raw-input signature so a served endpoint can
    enforce the same schema the pipeline was fit on. When ``register_name``
    is supplied the logged model is registered and moved to
    ``register_stage`` (for example ``Staging``).
    """
    mlflow = _require_mlflow()
    from mlflow.models import infer_signature

    configure_tracking(tracking_uri, experiment)

    folder = Path(folder)
    saved_pipeline_path = Path(saved_pipeline_path)
    metadata = result.metadata
    scope = data_scope or metadata.get("data_scope", "unrecorded")

    params = {
        "stage": "finalize",
        "artifact_id": metadata["artifact_id"],
        "artifact_role": metadata.get("artifact_role", ""),
        "model": metadata["model"],
        "encoding": metadata.get("encoding", ""),
        "selection_method": metadata.get("selection_method", "basic"),
        "feature_groups": metadata.get("feature_groups", "all"),
        "n_clusters": metadata.get("n_clusters", 0),
        "raw_screen_strategy": metadata.get("raw_screen_strategy", "all"),
        "top_v_features": metadata.get("top_v_features"),
        "threshold_objective": metadata.get("threshold_objective", ""),
        "threshold": result.threshold,
        "training_rows": metadata.get("training_rows", len(partitions.train)),
        "training_ids_hash": metadata.get("training_ids_hash", ""),
        "data_scope": scope,
    }

    run_id = None
    with mlflow.start_run(
        run_name=f"candidate_{metadata['artifact_id'][:8]}"
    ) as run:
        run_id = run.info.run_id

        mlflow.log_params({k: _to_param_text(v) for k, v in params.items()})
        _log_common_lineage(
            mlflow,
            metadata,
            extra_tags={
                "winner_name": comparison.winner_name if comparison else "single",
                "comparison_id": (
                    comparison.metadata["comparison_id"] if comparison else ""
                ),
                "data_scope": scope,
            },
        )

        # Per-partition metrics from the already evaluated result.
        for partition_name, partition_metrics in result.metrics.items():
            _log_scalar_metrics(mlflow, partition_name, partition_metrics)

        # Raw-input signature: the same schema a served endpoint would see.
        sample = partitions.validation.drop(columns="isFraud").head(5)
        try:
            signature = infer_signature(
                sample,
                result.pipeline.predict_proba(sample),
            )
        except Exception:  # signature is best-effort
            signature = None

        mlflow.sklearn.log_model(
            sk_model=result.pipeline,
            artifact_path="model",
            signature=signature,
            input_example=sample.head(1),
        )

        # Attach local report artifacts that already sit next to the model.
        for filename in (
            "metadata.json",
            "model_card.json",
            "holdout_slices.csv",
            "classifier_importance.csv",
        ):
            path = folder / filename
            if path.is_file():
                mlflow.log_artifact(str(path))

        if register_name:
            model_uri = f"runs:/{run_id}/model"
            registered = mlflow.register_model(
                model_uri=model_uri, name=register_name
            )
            if register_stage:
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=register_name,
                    version=registered.version,
                    stage=register_stage,
                )

    return run_id


def verify_registered_lineage(run_id, partitions, *, tracking_uri=None):
    """Confirm a stored run was fit on the current partitions.

    Reads the ``partition_fingerprints`` tag written by ``log_result`` and
    compares it against the fingerprints of ``partitions``. Call this before
    promoting a model version to ``Production`` so MLflow and the local
    lineage agree on the training data.
    """
    mlflow = _require_mlflow()
    from .utils import frame_fingerprint

    configure_tracking(tracking_uri)

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    stored_raw = run.data.tags.get("partition_fingerprints")
    if not stored_raw:
        raise ValueError(
            "Stored run has no partition_fingerprints tag; cannot verify lineage"
        )

    stored = json.loads(stored_raw)
    expected = {
        name: frame_fingerprint(getattr(partitions, name))
        for name in ("train", "validation", "holdout")
    }

    if stored != expected:
        raise ValueError(
            "Registered run lineage does not match the current partitions"
        )

    return expected