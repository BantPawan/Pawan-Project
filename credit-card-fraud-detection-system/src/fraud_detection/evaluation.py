"""Explain and document the fitted artifact after model selection.

Reports include aggregate error slices, native feature importance, optional
SHAP explanations and a model card. Every report checks the artifact's data
lineage and saved threshold before using it. No diagnostic helper fits
another model.

These diagnostics never fit or select models. Inspecting a final holdout to
choose another feature or model would turn that holdout into development data.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from .models import validate_partitions
from .utils import calculate_metrics, frame_fingerprint


def _check_lineage(result, partitions):
    """Verify that report rows, metrics and threshold belong to this artifact."""
    validate_partitions(partitions)

    expected = {
        name: frame_fingerprint(getattr(partitions, name))
        for name in ("train", "validation", "holdout")
    }
    if expected != result.metadata["partition_fingerprints"]:
        raise ValueError("Diagnostic rows differ from the evaluated artifact lineage")

    if result.metadata["artifact_id"] != result.metadata["metric_artifact_id"]:
        raise ValueError("Diagnostic metrics belong to another artifact")

    if (
        result.threshold != result.pipeline.threshold
        or result.threshold != result.metadata["threshold"]
    ):
        raise ValueError("Diagnostic threshold differs from the evaluated policy")


def error_analysis(result, partitions, partition="holdout"):
    """Report aggregate time, amount and identity slices without raw rows.

    Amount cuts come from training statistics. Time slices describe thirds of
    this assessment period and do not claim calendar dates. Slices with a
    single class or very few rows keep their counts and truthful uncomputable
    metrics rather than a fake AUC.
    """
    _check_lineage(result, partitions)

    if partition not in result.metrics:
        raise ValueError("Error analysis requires an already evaluated partition")

    frame = getattr(partitions, partition)
    raw = frame.drop(columns="isFraud")

    probability = result.pipeline.predict_proba(raw)[:, 1]
    labels = result.pipeline.predict(raw)

    stats = result.pipeline.pipeline.named_steps["features"].amount_stats_
    # Reuse training quantiles. The assessment period does not set amount cuts.
    lower_amounts = frame.TransactionAmt.le(stats["q25"])
    middle_amounts = frame.TransactionAmt.gt(stats["q25"]) & frame.TransactionAmt.le(
        stats["q75"]
    )
    higher_amounts = frame.TransactionAmt.gt(stats["q75"])

    groups = [
        ("overall", "all", np.ones(len(frame), dtype=bool)),
        ("amount", "at_or_below_training_q25", lower_amounts.to_numpy()),
        ("amount", "between_training_q25_q75", middle_amounts.to_numpy()),
        ("amount", "above_training_q75", higher_amounts.to_numpy()),
    ]

    times = frame.TransactionDT.to_numpy()
    unique_times = np.unique(times)
    time_windows = np.array_split(unique_times, min(3, len(unique_times)))
    for number, window in enumerate(time_windows):
        groups.append(
            ("elapsed_time_window", f"window_{number + 1}", np.isin(times, window))
        )

    fitted = result.pipeline.pipeline.named_steps["features"].input_columns_
    identity = [
        name
        for name in fitted
        if name.startswith("id_") or name in ("DeviceInfo", "DeviceType")
    ]
    if identity:
        present = raw.reindex(columns=identity).notna().any(axis=1).to_numpy()
        groups.extend(
            [
                ("identity", "observed", present),
                ("identity", "all_missing", ~present),
            ]
        )

    actual_labels = frame.isFraud.to_numpy()
    rows = []
    for dimension, group, mask in groups:
        if not mask.any():
            continue

        metrics = calculate_metrics(
            actual_labels[mask], labels[mask], probability[mask]
        )
        report_metrics = {
            key: value for key, value in metrics.items() if key != "confusion_matrix"
        }

        rows.append(
            {
                "partition": partition,
                "dimension": dimension,
                "slice": group,
                "positive_labels": int(actual_labels[mask].sum()),
                "small_slice": bool(mask.sum() < 30),
                **report_metrics,
            }
        )

    return pd.DataFrame(rows)


def candidate_feature_importance(result):
    """Explain the fitted classifier in its transformed feature space.

    Linear coefficients use the fitted scaling. Tree importances can favour
    high-cardinality variables. Neither proves that a feature causes fraud.
    """
    estimator = result.pipeline.pipeline.named_steps["classifier"]
    names = result.pipeline.pipeline.named_steps["selection"].selected_features_

    if hasattr(estimator, "coef_"):
        importance = np.abs(estimator.coef_).mean(axis=0)
        kind = "absolute_fitted_linear_coefficient"

    elif hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
        kind = "fitted_tree_importance"

    elif hasattr(estimator, "estimators_"):
        # Coefficients and tree importances have different units. Percentile
        # ranks give a descriptive member consensus, not an ensemble attribution.
        member_ranks = []
        for member in estimator.estimators_:
            if hasattr(member, "coef_"):
                values = np.abs(member.coef_).mean(axis=0)
            elif hasattr(member, "feature_importances_"):
                values = member.feature_importances_
            else:
                raise ValueError(
                    "An ensemble member has no supported native importance"
                )
            member_ranks.append(
                pd.Series(values).rank(method="average", pct=True).to_numpy()
            )
        importance = np.mean(member_ranks, axis=0)
        kind = "mean_member_importance_percentile_rank_descriptive"

    else:
        raise ValueError("This classifier has no supported native feature importance")

    report = pd.DataFrame(
        {
            "feature": names,
            "importance": importance,
            "kind": kind,
        }
    )
    return report.sort_values(
        ["importance", "feature"], ascending=[False, True]
    ).reset_index(drop=True)


def explain_candidate_shap(result, partitions, max_rows=128, seed=42):
    """Bounded SHAP of the selected classifier, using training rows only.

    The frame explains transformed features, including encoded categories.
    This is a descriptive explanation of the fitted model, not a new selection
    step.
    """
    _check_lineage(result, partitions)

    if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")

    try:
        import shap
    except ImportError as error:
        raise ImportError(
            "Install requirements-local.txt for SHAP explanations"
        ) from error

    raw = partitions.train.drop(columns="isFraud")
    estimator = result.pipeline.pipeline.named_steps["classifier"]
    ensemble = hasattr(estimator, "estimators_") and not hasattr(
        estimator, "feature_importances_"
    )

    # Model-agnostic ensemble explanations are deliberately bounded: explaining
    # 16 rows teaches the method without pretending to be exhaustive.
    sample_size = min(max_rows, 16 if ensemble else max_rows, len(raw))
    random = np.random.default_rng(seed)
    positions = np.sort(random.choice(len(raw), sample_size, replace=False))
    values = result.pipeline.transform(raw.iloc[positions])

    if hasattr(estimator, "coef_"):
        explanations = shap.LinearExplainer(estimator, values).shap_values(values)
        output_space = "linear classifier log-odds"

    elif ensemble:
        names = values.columns.tolist()
        background = values.iloc[: min(8, len(values))]

        def fraud_probability(matrix):
            frame = pd.DataFrame(matrix, columns=names)
            return estimator.predict_proba(frame)[:, 1]

        previous_state = np.random.get_state()
        try:
            np.random.seed(seed)
            explainer = shap.KernelExplainer(fraud_probability, background)
            explanations = explainer.shap_values(
                values, nsamples=64, l1_reg="num_features(10)", silent=True
            )
        finally:
            np.random.set_state(previous_state)
        output_space = (
            "ensemble fraud score probability; approximate Kernel SHAP "
            "(64 samples, 8 background rows maximum)"
        )

    else:
        explanations = shap.TreeExplainer(estimator).shap_values(values)
        output_space = (
            "classifier-specific raw output; not uniformly probability units"
        )

    if isinstance(explanations, list):
        explanations = explanations[-1]  # fraud class for older RF SHAP API
    explanations = np.asarray(explanations)
    if explanations.ndim == 3:
        explanations = explanations[:, :, 1]  # SHAP 0.46 binary RF class axis

    names = result.pipeline.pipeline.named_steps["selection"].selected_features_
    report = pd.DataFrame(
        {
            "feature": names,
            "mean_absolute_shap": np.abs(explanations).mean(axis=0),
        }
    )
    ranked_importance = report.sort_values(
        ["mean_absolute_shap", "feature"], ascending=[False, True]
    ).reset_index(drop=True)

    return {
        "importance": ranked_importance,
        "scope": "training_rows_only_descriptive_explanation",
        "explained_rows": len(positions),
        "sample_positions": positions.tolist(),
        "output_space": output_space,
        "shap_values": explanations,
        "sample_values": values,
    }


def save_shap_beeswarm(explanation, output_path, max_display=20):
    """Show the distribution and direction hidden by mean-absolute SHAP bars.

    This is a local diagnostic of training rows. Colour describes transformed
    feature values, so an encoded category's colour does not imply numeric
    order.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    values = np.asarray(explanation["shap_values"])
    sample = explanation["sample_values"]
    if values.shape != sample.shape:
        raise ValueError("SHAP values do not match the explained feature matrix")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing SHAP plot")

    figure = plt.figure(figsize=(9, 6))
    try:
        shap.summary_plot(values, sample, max_display=max_display, show=False)
        plt.title("Training-only SHAP distribution of the frozen winner")
        plt.tight_layout()
        plt.savefig(path, dpi=130, bbox_inches="tight")
    finally:
        plt.close(figure)

    return path


def model_card(result, partitions, data_scope=None):
    """Describe the assessed candidate, its evidence and its practical limits.

    Data scope separates generated checks from genuine-data experiments. A
    successful local assessment does not establish production readiness.
    """
    _check_lineage(result, partitions)

    if "holdout" not in result.metrics:
        raise ValueError("Model report requires final holdout assessment")

    fixture = any(
        getattr(partitions, name).attrs.get("data_scope") == "generated_fixture"
        for name in ("train", "validation", "holdout")
    )

    if data_scope:
        scope = data_scope
    elif result.metadata.get("data_scope"):
        scope = result.metadata["data_scope"]
    elif fixture:
        scope = "generated_fixture"
    else:
        scope = "unrecorded"

    if fixture and scope not in ("generated_fixture", "smoke"):
        raise ValueError("Generated fixtures cannot be reported as real model evidence")

    if scope not in (
        "generated_fixture",
        "smoke",
        "real",
        "real_sample",
        "real_full",
        "unrecorded",
    ):
        raise ValueError("Unknown model report data scope")

    if result.metadata.get("data_scope") not in (None, scope):
        raise ValueError("Model report data scope differs from recorded lineage")

    if scope in ("generated_fixture", "smoke"):
        validation_status = "fixture_checks_only"
    elif scope == "unrecorded":
        validation_status = "scope_unrecorded"
    else:
        validation_status = "local_holdout_evaluated_candidate"

    features = result.pipeline.pipeline.named_steps["features"]
    required_fields = ["TransactionID", "TransactionDT", "TransactionAmt"]
    optional_fields = [
        column
        for column in result.metadata["raw_input_columns"]
        if column not in required_fields
    ]

    return {
        "artifact_id": result.metadata["artifact_id"],
        "data_scope": scope,
        "validation_status": validation_status,
        "production_ready": False,
        "intended_use": (
            "local learning and candidate evaluation on joined raw transactions"
        ),
        "model": result.metadata["model"],
        "encoding": getattr(features, "encoding_", "legacy"),
        "feature_groups": list(getattr(features, "feature_groups_", ())),
        "selection_method": result.metadata.get("selection_method", "basic"),
        "selection_criterion": result.metadata.get(
            "selection_criterion", "explicit single candidate"
        ),
        "threshold": result.threshold,
        "threshold_objective": result.metadata["threshold_objective"],
        "fit_partitions": result.metadata["fit_partitions"],
        "training_rows": result.metadata["training_rows"],
        "metrics": result.metrics,
        "raw_input_contract": {
            "required": required_fields,
            "optional_fitted_fields": optional_fields,
        },
        "splits": result.metadata["split_report"],
        "limitations": [
            "Generated fixtures establish software behaviour only; genuine performance requires genuine data.",
            "Threshold maximizes validation F1; business false-alarm and missed-fraud costs were not supplied.",
            "Class-weighted scores are not established as calibrated fraud probabilities; Brier score is diagnostic.",
            "Frequency features are frozen training lookups, not causal customer history.",
            "CV and tuning scores are development estimates; the selected configuration lacks nested-CV evaluation.",
            "Repeated decisions based on inspected holdout results require a new final assessment period.",
            "No serving, live monitoring or automatic retraining is implemented in this local phase.",
        ],
    }


def save_evaluation_reports(result, partitions, output_dir, data_scope=None):
    """Write aggregate diagnostics beside the exact model, without raw rows."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    card = model_card(result, partitions, data_scope)
    card_path = directory / "model_card.json"
    slices_path = directory / "holdout_slices.csv"
    importance_path = directory / "classifier_importance.csv"

    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    error_analysis(result, partitions).to_csv(slices_path, index=False)
    candidate_feature_importance(result).to_csv(importance_path, index=False)

    return {
        "model_card": str(card_path),
        "holdout_slices": str(slices_path),
        "classifier_importance": str(importance_path),
    }


def save_comparison_plots(comparison, labeled_validation, output_dir):
    """Plot the validation evidence used to rank candidates.

    The input fingerprint must match the comparison's validation frame.
    Plotting another period under these labels would misrepresent the
    selection evidence.
    """
    expected = comparison.metadata["partition_fingerprints"]["validation"]
    if frame_fingerprint(labeled_validation) != expected:
        raise ValueError(
            "Comparison plots require the same validation rows used for ranking"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    figures = []

    raw = labeled_validation.drop(columns="isFraud")
    target = labeled_validation.isFraud

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, result in comparison.results.items():
        probability = result.pipeline.predict_proba(raw)[:, 1]
        RocCurveDisplay.from_predictions(target, probability, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(
            target, probability, name=name, ax=axes[1]
        )
    axes[0].set_title("Validation ROC — development data")
    axes[1].set_title("Validation precision/recall — development data")
    fig.tight_layout()
    path = directory / "validation_curves.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append(path)

    table = comparison.table.sort_values("rank", ascending=False)
    fig, axis = plt.subplots(figsize=(8, max(3, 0.4 * len(table))))
    axis.barh(table.candidate, table.validation_average_precision)
    axis.set(
        xlabel="Validation average precision",
        title=f"{comparison.metadata['data_scope']} comparison",
    )
    fig.tight_layout()
    path = directory / "validation_average_precision.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append(path)

    return figures