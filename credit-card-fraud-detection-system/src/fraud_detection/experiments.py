"""Compare complete pipelines and preserve the selected model for assessment.

The workflow keeps three data roles separate:

1. Training contains the chronological folds used for screening and tuning.
2. Validation ranks the finished candidates and chooses decision thresholds.
3. Holdout provides a final assessment of the selected, unchanged winner.

All search folds live inside the earlier training period. Validation ranks the
finished candidates and selects their decision thresholds. Only the frozen
winner can receive a final holdout assessment. No helper implicitly refits it.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import PROJECT_ROOT, validate_transactions
from .feature_selection import DEFAULT_CONSENSUS_COMPONENTS
from .models import (
    FraudPipeline,
    TrainingResult,
    build_classifier,
    evaluate_pipeline,
    parameter_record,
    train_candidate,
    validate_partitions,
)
from .utils import frame_fingerprint, column_view


SELECTION_CRITERION = "validation_average_precision"
MODELS = ("logistic", "random_forest", "xgboost", "lightgbm", "ensemble")


@dataclass
class ComparisonResult:
    """Keep candidate objects, ranking evidence and the winner in one record.

    ``results`` holds fitted pipelines rather than settings for later refitting.
    Saving this record lets the assessment notebook load the selected artifact
    even when it runs in a different Python kernel.
    """

    results: dict
    table: pd.DataFrame
    winner_name: str
    cv_results: list
    tuning_results: list
    skipped: dict
    metadata: dict
    screening_comparison: object = None

    @property
    def winner(self):
        """Return the fitted result selected by the stored validation ranking."""
        return self.results[self.winner_name]


@dataclass
class FeatureSetStudy:
    """Keep paired feature experiments separate for each classifier family.

    The combined table is a learning report. It does not select a final model
    across families or authorize holdout assessment. Each comparison contains
    its own fitted candidates and complete chronological fold evidence.
    """

    comparisons: dict
    table: pd.DataFrame
    model_status: pd.DataFrame
    skipped: dict
    metadata: dict


def default_candidates(smoke=False, gpu=False):
    """The original four classifier families with bounded estimator settings.

    Smoke changes estimator count, not the comparison's selection rules. Generated
    scores demonstrate integration only; they never establish genuine performance.
    GPU mode uses CUDA for XGBoost and OpenCL for LightGBM; the other
    classifier families keep their existing CPU backends.
    """
    trees = 24 if smoke else 80
    return {
        "logistic": {"model": "logistic", "encoding": "onehot"},
        "random_forest": {
            "model": "random_forest",
            "encoding": "legacy",
            "model_params": {"n_estimators": trees},
        },
        "xgboost": {
            "model": "xgboost",
            "encoding": "legacy",
            "model_params": {
                "n_estimators": trees,
                **({"device": "cuda"} if gpu else {}),
            },
        },
        "lightgbm": {
            "model": "lightgbm",
            "encoding": "legacy",
            "model_params": {
                "n_estimators": trees,
                **({"device_type": "gpu"} if gpu else {}),
            },
        },
    }


def chronological_cv_splits(labeled_train, n_splits=3):
    """Expanding windows over distinct timestamp groups, preserving time ties.

    Returned indices refer to the supplied frame, including shuffled caller rows.
    Durations need not be equal: these are group-based diagnostic folds rather
    than a claim that the transaction stream is sampled at a fixed frequency.
    """
    validate_transactions(labeled_train, labeled=True)
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    times = labeled_train.TransactionDT.to_numpy()
    groups = np.unique(times)
    if len(groups) < n_splits + 1:
        raise ValueError("Too few distinct training timestamps for chronological folds")
    windows = np.array_split(groups, n_splits + 1)
    folds = []
    reports = []
    for number in range(n_splits):
        past = np.concatenate(windows[: number + 1])
        future = windows[number + 1]
        fit = np.flatnonzero(np.isin(times, past))
        assessment = np.flatnonzero(np.isin(times, future))
        fit = fit[np.argsort(times[fit], kind="stable")]
        assessment = assessment[np.argsort(times[assessment], kind="stable")]
        for role, positions in (("training", fit), ("assessment", assessment)):
            if set(labeled_train.iloc[positions].isFraud.unique()) != {0, 1}:
                raise ValueError(
                    f"Chronological fold {number + 1} {role} lacks both classes; "
                    "use more genuine rows or fewer folds"
                )
        earlier = labeled_train.iloc[fit]
        later = labeled_train.iloc[assessment]
        folds.append((fit, assessment))
        reports.append(
            {
                "fold": number + 1,
                "fit_rows": len(fit),
                "assessment_rows": len(assessment),
                "fit_elapsed_max": float(earlier.TransactionDT.max()),
                "assessment_elapsed_min": float(later.TransactionDT.min()),
                "fit_ids_hash": frame_fingerprint(earlier[["TransactionID"]]),
                "assessment_ids_hash": frame_fingerprint(later[["TransactionID"]]),
                "fit_class_counts": {
                    str(i): int(earlier.isFraud.eq(i).sum()) for i in (0, 1)
                },
                "assessment_class_counts": {
                    str(i): int(later.isFraud.eq(i).sum()) for i in (0, 1)
                },
                "fit_data_fingerprint": frame_fingerprint(earlier),
                "assessment_data_fingerprint": frame_fingerprint(later),
            }
        )
    return folds, reports


def _normalize_config(config):
    """Validate settings and copy them so later caller edits cannot change them."""
    allowed = {
        "model",
        "encoding",
        "selection_method",
        "selection_max_features",
        "selection_consensus_components",
        "selection_device",
        "feature_groups",
        "n_clusters",
        "model_params",
        "raw_screen_strategy",
        "top_v_features",
        "preserve_features",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown candidate configuration fields: {sorted(unknown)}")
    if config.get("model") not in MODELS:
        raise ValueError(f"Candidate model must be one of {MODELS}")
    result = {
        "encoding": "auto",
        "selection_method": "basic",
        "selection_max_features": None,
        "selection_consensus_components": DEFAULT_CONSENSUS_COMPONENTS,
        "selection_device": "cpu",
        "feature_groups": "all",
        "n_clusters": 0,
        "model_params": {},
        "raw_screen_strategy": "all",
        "top_v_features": None,
        "preserve_features": (),
        **deepcopy(config),
    }
    result["model_params"] = dict(result["model_params"] or {})
    if isinstance(result["feature_groups"], list):
        result["feature_groups"] = tuple(result["feature_groups"])
    if not isinstance(result["preserve_features"], (tuple, list)):
        raise ValueError("preserve_features must be a tuple or list of feature names")
    result["preserve_features"] = tuple(result["preserve_features"])
    if not isinstance(result["selection_consensus_components"], (tuple, list)):
        raise ValueError("selection_consensus_components must be a tuple or list")
    result["selection_consensus_components"] = tuple(result["selection_consensus_components"])
    if result["selection_device"] not in ("cpu", "gpu"):
        raise ValueError("selection_device must be cpu or gpu")
    return result


def _json_default(value):
    """Convert NumPy and path values used in local comparison reports."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, pd.Index)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _cv_score(labeled_train, config, seed, folds, fold_reports):
    """Refit the entire pipeline on each past window and score its next window.

    A fresh pipeline is important: encoding maps, amount statistics, selectors
    and medians must be learned from that fold's past rows. Fold reports retain
    enough evidence to check which rows produced the fitted statistics.
    """
    raw = column_view(labeled_train, [c for c in labeled_train if c != "isFraud"])
    labels = labeled_train.isFraud
    scores = []
    for (fit, assessment), report in zip(folds, fold_reports):
        started = time.perf_counter()
        pipeline = FraudPipeline(seed=seed, **config)
        pipeline.fit(raw.iloc[fit], labels.iloc[fit])
        elapsed = time.perf_counter() - started
        probability = pipeline.predict_proba(raw.iloc[assessment])[:, 1]
        features = pipeline.pipeline.named_steps["features"]
        selector = pipeline.pipeline.named_steps["selection"]
        assessment_labels = labels.iloc[assessment]
        scores.append(
            {
                **report,
                "average_precision": float(
                    average_precision_score(assessment_labels, probability)
                ),
                "roc_auc": float(roc_auc_score(assessment_labels, probability)),
                "fit_seconds": elapsed,
                "training_ids_hash": pipeline.training_ids_hash_,
                "fitted_amount_mean": features.amount_stats_["mean"],
                "selected_feature_count": len(selector.selected_features_),
                "encoding": pipeline.encoding_,
                "selection_method": config["selection_method"],
                "selection_device": config["selection_device"],
                "selection_consensus_components": list(config["selection_consensus_components"]),
                "raw_screen_strategy": config["raw_screen_strategy"],
                "raw_screen_fit_rows": len(raw.iloc[fit]),
                "raw_selected_feature_count": len(
                    pipeline.pipeline.named_steps["raw_screen"].selected_features_
                ),
            }
        )

    average_precision_scores = [report["average_precision"] for report in scores]
    roc_auc_scores = [report["roc_auc"] for report in scores]
    return {
        "mean_average_precision": float(np.mean(average_precision_scores)),
        "std_average_precision": float(np.std(average_precision_scores)),
        "mean_roc_auc": float(np.mean(roc_auc_scores)),
        "folds": scores,
    }


def _tuning_overrides(model):
    """Return two modest parameter trials for a promising classifier family."""
    # Search changes one interpretable parameter per model; the outer candidate
    # configuration controls feature/encoding experiments independently.
    return {
        "logistic": [{"C": 0.5}, {"C": 2.0}],
        "random_forest": [{"max_depth": 6}, {"max_depth": 12}],
        "xgboost": [{"max_depth": 3}, {"max_depth": 5}],
        "lightgbm": [{"num_leaves": 7}, {"num_leaves": 31}],
        "ensemble": [],
    }[model]


def _summary(comparison):
    """Create the readable report that accompanies fitted comparison objects."""
    return {
        **comparison.metadata,
        "winner_name": comparison.winner_name,
        "winner_artifact_id": comparison.winner.metadata["artifact_id"],
        "table": comparison.table.to_dict(orient="records"),
        "cv_results": comparison.cv_results,
        "tuning_results": comparison.tuning_results,
        "skipped": comparison.skipped,
        "holdout_policy": "only frozen winner assessed; no loser holdout scores",
    }


def compare_candidates(
    partitions,
    candidates=None,
    seed=42,
    n_splits=3,
    tune=True,
    evaluate_holdout=False,
    data_scope=None,
    tune_top_k=2,
):
    """Select by validation average precision after training-only CV and tuning.

    First, each candidate runs through expanding training folds. The strongest
    ``tune_top_k`` candidates may try two bounded parameter alternatives on the
    same training folds. Each resulting configuration then fits the full training
    partition; validation provides its threshold and ranking evidence.

    Ties prefer validation ROC-AUC then alphabetical candidate name. Training CV
    screens tuning recipients; it is a development estimate, not nested-CV proof
    for the selected configuration. Runtime never breaks ties. Missing optional
    boosting dependencies are reported explicitly; other training failures raise.
    """
    validate_partitions(partitions)
    if (
        not isinstance(tune_top_k, int)
        or isinstance(tune_top_k, bool)
        or tune_top_k < 1
    ):
        raise ValueError("tune_top_k must be a positive integer")
    fixture = any(
        getattr(partitions, name).attrs.get("data_scope") == "generated_fixture"
        for name in ("train", "validation", "holdout")
    )
    if data_scope:
        scope = data_scope
    else:
        scope = "generated_fixture" if fixture else "real"
    if scope not in ("generated_fixture", "smoke", "real", "real_sample", "real_full"):
        raise ValueError("Unknown comparison data scope")
    if fixture and scope not in ("generated_fixture", "smoke"):
        raise ValueError("Generated fixtures cannot be compared as real data")
    if candidates is None:
        proposed = default_candidates(smoke=fixture)
    else:
        proposed = candidates
    if not proposed or any(not isinstance(name, str) or not name for name in proposed):
        raise ValueError("At least one named candidate is required")
    configs = {name: _normalize_config(config) for name, config in proposed.items()}

    # Stage 1: screen every candidate using folds inside the training period.
    folds, fold_reports = chronological_cv_splits(partitions.train, n_splits)
    cv = {}
    cv_records = []
    tuning_records = []
    skipped = {}
    for name, config in configs.items():
        try:
            score = _cv_score(partitions.train, config, seed, folds, fold_reports)
        except ImportError as error:
            skipped[name] = str(error)
            continue
        cv[name] = score
        cv_records.append(
            {"candidate": name, "phase": "initial", "config": config, **score}
        )
    if not cv:
        raise ValueError(f"No candidates could run: {skipped}")

    # Stage 2: tune a bounded number of candidates using training folds again.
    # Validation and holdout labels have no role in selecting these settings.
    effective = {name: deepcopy(configs[name]) for name in cv}
    promising = sorted(
        cv,
        key=lambda name: (
            -cv[name]["mean_average_precision"],
            -cv[name]["mean_roc_auc"],
            name,
        ),
    )[:tune_top_k]
    if tune:
        for name in promising:
            options = [(effective[name], cv[name])]
            for overrides in _tuning_overrides(effective[name]["model"]):
                config = {
                    **effective[name],
                    "model_params": {**effective[name]["model_params"], **overrides},
                }
                score = _cv_score(partitions.train, config, seed, folds, fold_reports)
                options.append((config, score))
                tuning_records.append({"candidate": name, "config": config, **score})
            options.sort(
                key=lambda option: (
                    -option[1]["mean_average_precision"],
                    -option[1]["mean_roc_auc"],
                    json.dumps(option[0]["model_params"], sort_keys=True),
                )
            )
            effective[name], cv[name] = options[0]
    # Stage 3: fit final candidates on training, then rank validation evidence.
    results = {}
    rows = []
    for name, config in effective.items():
        started = time.perf_counter()
        result = train_candidate(
            partitions,
            seed=seed,
            evaluate_holdout=False,
            **deepcopy(config),
        )
        elapsed = time.perf_counter() - started
        result.metadata.update(
            candidate_name=name, data_scope=scope, candidate_config=deepcopy(config)
        )
        results[name] = result
        validation_metrics = {
            f"validation_{key}": value
            for key, value in result.metrics["validation"].items()
            if key != "confusion_matrix"
        }
        rows.append(
            {
                "candidate": name,
                "model": config["model"],
                "encoding": result.pipeline.encoding_,
                "selection_method": config["selection_method"],
                "selection_device": config["selection_device"],
                "feature_groups": config["feature_groups"],
                "n_clusters": config["n_clusters"],
                "model_params": config["model_params"],
                "cv_average_precision": cv[name]["mean_average_precision"],
                "cv_average_precision_std": cv[name]["std_average_precision"],
                "cv_roc_auc": cv[name]["mean_roc_auc"],
                **validation_metrics,
                "threshold": result.threshold,
                "fit_seconds": elapsed,
                "selected_feature_count": len(result.metadata["selected_features"]),
            }
        )
    table = (
        pd.DataFrame(rows)
        .sort_values(
            ["validation_average_precision", "validation_roc_auc", "candidate"],
            ascending=[False, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    winner = str(table.iloc[0].candidate)
    table["selected"] = table.candidate.eq(winner)
    # Stage 4: freeze the ranking, settings and data fingerprints for handoff.
    metadata = {
        "comparison_id": uuid.uuid4().hex,
        "data_scope": scope,
        "seed": seed,
        "n_splits": n_splits,
        "selection_criterion": SELECTION_CRITERION,
        "tie_policy": "validation ROC-AUC descending, then candidate name ascending",
        "tuning_criterion": "training chronological CV average precision",
        "tuned_candidates": promising if tune else [],
        "tune_top_k": tune_top_k,
        "requested_configs": deepcopy(configs),
        "effective_configs": deepcopy(effective),
        "fold_boundaries": fold_reports,
        "partition_fingerprints": {
            name: frame_fingerprint(getattr(partitions, name))
            for name in ("train", "validation", "holdout")
        },
        "holdout_evaluated": False,
    }
    comparison = ComparisonResult(
        results, table, winner, cv_records, tuning_records, skipped, metadata
    )
    comparison.winner.metadata["comparison_summary"] = _summary(comparison)
    comparison.winner.metadata["selection_criterion"] = SELECTION_CRITERION
    if evaluate_holdout:
        evaluate_selected_comparison(comparison, partitions)
    return comparison


def shortlist_candidates(screening_comparison, shortlist_size=3):
    """Promote candidates using their initial training CV evidence only.

    Validation scores are deliberately absent from this decision. This helper
    accepts an untuned, unassessed screening comparison so the detailed study
    can consume an existing notebook handoff without repeating Phase A.
    """
    if (
        not isinstance(shortlist_size, int)
        or isinstance(shortlist_size, bool)
        or shortlist_size < 1
    ):
        raise ValueError("shortlist_size must be a positive integer")
    if screening_comparison.tuning_results:
        raise ValueError(
            "Phase A must be untuned; reserve tuning for the detailed study"
        )
    if screening_comparison.metadata.get("holdout_evaluated") or any(
        "holdout" in result.metrics for result in screening_comparison.results.values()
    ):
        raise ValueError("Phase A must not contain holdout assessments")
    scores = {
        record["candidate"]: record
        for record in screening_comparison.cv_results
        if record.get("phase") == "initial"
    }
    if set(scores) != set(screening_comparison.results):
        raise ValueError(
            "Phase A lacks initial training CV evidence for every candidate"
        )
    ordered = sorted(
        scores,
        key=lambda name: (
            -scores[name]["mean_average_precision"],
            -scores[name]["mean_roc_auc"],
            name,
        ),
    )
    return ordered[:shortlist_size]


def staged_model_study(
    partitions,
    candidates=None,
    seed=42,
    screening_folds=2,
    detail_folds=3,
    shortlist_size=3,
    detail_estimators=160,
    tune=True,
    data_scope=None,
    include_ensemble=True,
    screening_comparison=None,
):
    """Run visible screening, detailed evaluation, tuning and ensemble stages.

    Phase A uses earlier training folds to shortlist model configurations.
    Phase B refits those complete raw-row pipelines on a larger fold budget and
    the declared tree budget. Bounded hyperparameter trials also use training
    folds. Validation then ranks final candidates and sets their thresholds.

    An optional soft-voting candidate uses fixed equal weights, common one-hot
    features and selective scaling. Its members are freshly fitted classifiers;
    the weights never depend on validation AUC or holdout outcomes. It is a new
    ensemble experiment, not an average of differently transformed saved models.
    Holdout remains reserved for ``evaluate_selected_comparison``.
    """
    validate_partitions(partitions)
    if (
        not isinstance(screening_folds, int)
        or isinstance(screening_folds, bool)
        or screening_folds < 2
    ):
        raise ValueError("screening_folds must be an integer of at least 2")
    if (
        not isinstance(shortlist_size, int)
        or isinstance(shortlist_size, bool)
        or shortlist_size < 1
    ):
        raise ValueError("shortlist_size must be a positive integer")
    if (
        not isinstance(detail_folds, int)
        or isinstance(detail_folds, bool)
        or detail_folds <= screening_folds
    ):
        raise ValueError("detail_folds must exceed screening_folds")
    if (
        not isinstance(detail_estimators, int)
        or isinstance(detail_estimators, bool)
        or detail_estimators < 1
    ):
        raise ValueError("detail_estimators must be a positive integer")

    if screening_comparison is None:
        # The screening helper fits all transformations within each CV fold.
        # Its validation table is visible context, but never selects the shortlist.
        screening = compare_candidates(
            partitions,
            candidates=candidates,
            seed=seed,
            n_splits=screening_folds,
            tune=False,
            evaluate_holdout=False,
            data_scope=data_scope,
        )
    else:
        screening = screening_comparison
        validate_comparison_contract(screening, partitions, data_scope=data_scope)
        if screening.metadata["n_splits"] != screening_folds:
            raise ValueError("Saved Phase A fold budget differs from screening_folds")
        if screening.metadata["seed"] != seed:
            raise ValueError("Saved Phase A seed differs from the detailed study")
        if candidates is not None:
            declared = {
                name: _normalize_config(config) for name, config in candidates.items()
            }
            # Saved CPU comparisons predate the explicit selection device.
            # Supply only that inert default; other recorded settings stay strict.
            recorded = {
                name: {"selection_device": "cpu", **config}
                for name, config in screening.metadata["requested_configs"].items()
            }
            if declared != recorded:
                raise ValueError("Saved Phase A candidate configurations differ")

    shortlist = shortlist_candidates(screening, shortlist_size)
    detailed = {}
    for name in shortlist:
        config = deepcopy(screening.metadata["effective_configs"][name])
        if config["model"] in ("random_forest", "xgboost", "lightgbm"):
            config["model_params"]["n_estimators"] = detail_estimators
        detailed[name] = config

    ensemble_policy = {"enabled": False, "reason": "ensemble stage disabled"}
    if include_ensemble and len(shortlist) >= 2:
        ensemble_name = "equal_weight_ensemble"
        if ensemble_name in detailed or any(
            config["model"] == "ensemble" for config in detailed.values()
        ):
            raise ValueError(
                "Phase A must contain individual models for ensemble study"
            )
        members = [
            {
                "name": name,
                "model": detailed[name]["model"],
                "model_params": deepcopy(detailed[name]["model_params"]),
            }
            for name in shortlist
        ]
        common_recipe = deepcopy(detailed[shortlist[0]])
        detailed[ensemble_name] = {
            **common_recipe,
            "model": "ensemble",
            "encoding": "onehot",
            "model_params": {"members": members},
        }
        ensemble_policy = {
            "enabled": True,
            "candidate": ensemble_name,
            "members": shortlist.copy(),
            "weights": [1.0 / len(members)] * len(members),
            "weight_source": "fixed equal weights declared before validation",
            "shared_recipe_source": shortlist[0],
        }
    elif include_ensemble:
        ensemble_policy["reason"] = "at least two available shortlisted models required"

    comparison = compare_candidates(
        partitions,
        candidates=detailed,
        seed=seed,
        n_splits=detail_folds,
        tune=tune,
        tune_top_k=len(detailed),
        evaluate_holdout=False,
        data_scope=screening.metadata["data_scope"],
    )
    phase_a = deepcopy(screening.cv_results)
    phase_b = deepcopy(comparison.cv_results)
    for record in phase_a:
        record["phase"] = "phase_a_screening"
    for record in phase_b:
        record["phase"] = "phase_b_detailed"
    comparison.cv_results = phase_a + phase_b
    # Keep Phase A models as actual artifacts, including candidates not promoted.
    # Their validation-only evidence can then be tracked beside Phase B runs.
    comparison.screening_comparison = screening
    comparison.metadata.update(
        workflow_type="staged_model_study",
        tuning_enabled=tune,
        screening_comparison_id=screening.metadata["comparison_id"],
        screening_winner_artifact_id=screening.winner.metadata["artifact_id"],
        shortlist=shortlist,
        shortlist_criterion="initial training chronological CV average precision",
        phase_a_results=phase_a,
        phase_a_validation_table=screening.table.to_dict(orient="records"),
        phase_b_results=phase_b,
        stage_budgets={
            "screening_folds": screening_folds,
            "detail_folds": detail_folds,
            "detail_estimators": detail_estimators,
            "shortlist_size": shortlist_size,
            "actual_shortlist_size": len(shortlist),
        },
        ensemble_policy=ensemble_policy,
        phase_a_skipped=deepcopy(screening.skipped),
        stage_order=[
            "phase_a_screening",
            "training_cv_shortlist",
            "phase_b_detailed",
            "training_cv_tuning",
            "fixed_weight_ensemble_comparison",
            "validation_ranking_and_thresholds",
            "freeze_for_holdout",
        ],
    )
    comparison.skipped = {**screening.skipped, **comparison.skipped}
    comparison.winner.metadata["comparison_summary"] = _summary(comparison)
    return comparison


def validate_study_handoff(
    comparison,
    screening,
    partitions,
    *,
    detail_folds,
    detail_estimators,
    shortlist_size,
    tune,
    data_scope=None,
):
    """Reject a stale detailed study when its baseline or declared budget changes.

    Reusing the saved winner is useful for revisiting the learning notebook. It
    must not silently present old settings under a newly requested experiment.
    A changed policy requires a new study and a fresh final assessment period.
    """
    validate_comparison_contract(screening, partitions, data_scope)
    validate_comparison_contract(comparison, partitions, data_scope)
    if (
        comparison.metadata.get("screening_comparison_id")
        != screening.metadata["comparison_id"]
    ):
        raise ValueError("Detailed study belongs to a different baseline comparison")
    expected_budget = {
        "screening_folds": screening.metadata["n_splits"],
        "detail_folds": detail_folds,
        "detail_estimators": detail_estimators,
        "shortlist_size": shortlist_size,
    }
    actual_budget = comparison.metadata.get("stage_budgets", {})
    if (
        any(actual_budget.get(name) != value for name, value in expected_budget.items())
        or comparison.metadata.get("tuning_enabled") != tune
    ):
        raise ValueError(
            "Saved detailed-study settings differ; start a new study with a fresh final assessment period"
        )
    if comparison.metadata.get("shortlist") != shortlist_candidates(
        screening, shortlist_size
    ):
        raise ValueError("Saved study shortlist differs from its training CV evidence")
    if comparison.metadata.get("seed") != screening.metadata["seed"]:
        raise ValueError("Saved study seed differs from its baseline")
    for field in ("feature_recipe", "feature_stage_checkpoint"):
        if screening.metadata.get(field) != comparison.metadata.get(field):
            raise ValueError(
                "Saved study feature-stage provenance differs from screening"
            )
    return comparison.winner


def compare_feature_sets(partitions, recipe, n_splits=2, seed=42, data_scope=None):
    """Compare full versus selected features using freshly fitted raw pipelines.

    Each candidate shares the supplied model and engineering recipe. ``full``
    retains all usable columns after basic quality filters; ``selected`` applies
    the requested ranking method. If business features are named, a third
    candidate tests that explicit preservation policy. No tuning or holdout
    assessment is performed, and precomputed selected matrices are never used.
    """
    selected = _normalize_config({"model": "random_forest", **deepcopy(recipe)})
    if selected["selection_method"] == "basic":
        raise ValueError(
            "Choose a supervised selection method for the selected comparison"
        )
    full = {
        **deepcopy(selected),
        "selection_method": "basic",
        "selection_max_features": None,
        "preserve_features": (),
    }
    unprotected = {**deepcopy(selected), "preserve_features": ()}
    candidates = {"full": full, "selected": unprotected}
    if selected["preserve_features"]:
        candidates["selected_preserving_business"] = selected
    comparison = compare_candidates(
        partitions,
        candidates=candidates,
        seed=seed,
        n_splits=n_splits,
        tune=False,
        evaluate_holdout=False,
        data_scope=data_scope,
    )
    comparison.metadata["workflow_type"] = "full_vs_selected_feature_study"
    comparison.metadata["selection_recipe"] = deepcopy(selected)
    comparison.winner.metadata["comparison_summary"] = _summary(comparison)
    return comparison


def compare_feature_sets_across_models(
    partitions,
    recipe,
    *,
    model_settings=None,
    n_splits=2,
    seed=42,
    data_scope=None,
):
    """Pair full, selected and business-preserved views for the original models.

    Random Forest, logistic regression and LightGBM each receive the same
    engineering and selection recipe. Their automatic encoding policy gives
    logistic regression one-hot categories and continuous scaling, while trees
    use category codes and unscaled values. Within each family every variant
    uses identical classifier settings and preprocessing. Each chronological
    fold starts from raw rows and fits its own complete pipeline.

    ``model_settings`` can override ``encoding`` and ``model_params`` per family
    to bound estimator work; it cannot silently change feature policies between
    variants. Missing LightGBM is reported explicitly. Missing requested
    selector dependencies raise, rather than dropping consensus components.
    There is no tuning, cross-family winner or holdout evaluation in this study.
    """
    families = ("random_forest", "logistic", "lightgbm")
    if model_settings is not None and not isinstance(model_settings, dict):
        raise ValueError("model_settings must map model families to their settings")
    settings = deepcopy(model_settings or {})
    unknown_families = set(settings) - set(families)
    if unknown_families:
        raise ValueError(f"Unknown paired-study model families: {sorted(unknown_families)}")
    base_recipe = deepcopy(recipe)
    base_recipe.pop("model", None)
    base_recipe.pop("model_params", None)
    comparisons = {}
    tables = []
    statuses = []
    skipped = {}
    configs = {}
    for model in families:
        overrides = settings.get(model, {})
        if not isinstance(overrides, dict) or set(overrides) - {"encoding", "model_params"}:
            raise ValueError("Each model setting may override only encoding and model_params")
        config = _normalize_config({
            **base_recipe,
            "model": model,
            "encoding": overrides.get("encoding", "auto"),
            "model_params": overrides.get("model_params", {}),
        })
        configs[model] = config
    for model, config in configs.items():
        # Check optional classifier availability before starting any variants.
        # Selector dependency failures inside the comparison are not skipped.
        try:
            build_classifier(model, seed, config["model_params"])
        except ImportError as error:
            skipped[model] = str(error)
            statuses.append({"model": model, "status": "skipped", "detail": str(error)})
            continue
        comparison = compare_feature_sets(
            partitions, config, n_splits=n_splits, seed=seed, data_scope=data_scope
        )
        if comparison.skipped:
            raise ImportError(
                f"Paired {model} feature study could not execute every variant: "
                f"{comparison.skipped}"
            )
        comparisons[model] = comparison
        tables.append(comparison.table.assign(study_model=model))
        statuses.append({"model": model, "status": "fitted", "detail": "all requested variants"})
    return FeatureSetStudy(
        comparisons=comparisons,
        table=pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(),
        model_status=pd.DataFrame(statuses, columns=["model", "status", "detail"]),
        skipped=skipped,
        metadata={
            "workflow_type": "paired_feature_sets_across_original_models",
            "data_scope": (
                next(iter(comparisons.values())).metadata["data_scope"]
                if comparisons else data_scope
            ),
            "seed": seed,
            "n_splits": n_splits,
            "requested_model_configs": configs,
            "holdout_evaluated": False,
            "tuning_performed": False,
            "cross_model_selection_performed": False,
            "metric_scope": "chronological_training_cv_and_reserved_validation",
        },
    )


def validate_comparison_contract(comparison, partitions, data_scope=None):
    """Verify that a saved comparison still identifies the same selected model.

    Fingerprints catch changed data; snapshots catch changed configuration or
    thresholds. Ranking is recomputed from its recorded validation evidence.
    These checks prevent assessment from silently using a different artifact.
    """
    validate_partitions(partitions)
    expected = {
        name: frame_fingerprint(getattr(partitions, name))
        for name in ("train", "validation", "holdout")
    }
    if expected != comparison.metadata["partition_fingerprints"]:
        raise ValueError(
            "Comparison handoff data fingerprints differ; rerun notebook 03"
        )
    if data_scope is not None and comparison.metadata["data_scope"] != data_scope:
        raise ValueError("Comparison handoff data scope differs")
    if comparison.metadata["selection_criterion"] != SELECTION_CRITERION:
        raise ValueError("Comparison selection criterion differs")
    if comparison.winner_name not in comparison.results:
        raise ValueError("Comparison handoff has no selected winner")
    rank = comparison.table.sort_values(
        ["validation_average_precision", "validation_roc_auc", "candidate"],
        ascending=[False, False, True],
    )
    if str(rank.iloc[0].candidate) != comparison.winner_name:
        raise ValueError("Comparison winner differs from declared ranking")
    for name, result in comparison.results.items():
        if result.metadata["partition_fingerprints"] != expected:
            raise ValueError("Candidate lineage differs from comparison handoff")
        config = comparison.metadata["effective_configs"][name]
        if result.metadata["candidate_config"] != config:
            raise ValueError("Candidate configuration differs from comparison handoff")
        actual = result.pipeline.get_params(deep=False)
        if any(actual[key] != value for key, value in config.items()):
            raise ValueError("Fitted candidate configuration changed after comparison")
        selection_device = config.get("selection_device", "cpu")
        selector = result.pipeline.pipeline.named_steps["selection"]
        if (
            selector.device != selection_device
            or result.metadata.get("selection_device", "cpu") != selection_device
        ):
            raise ValueError("Fitted selection device changed after comparison")
        classifier = result.pipeline.pipeline.named_steps["classifier"]
        classifier_parameters = parameter_record(classifier.get_params())
        if classifier_parameters != result.metadata["classifier_parameters"]:
            raise ValueError("Fitted classifier parameters changed after comparison")
        if result.pipeline.model == "ensemble":
            fitted_members = {
                member_name: parameter_record(member.get_params())
                for member_name, member in classifier.named_estimators_.items()
            }
            if fitted_members != result.metadata["ensemble_member_parameters"]:
                raise ValueError("Fitted ensemble member parameters changed")
        recorded_threshold = result.metadata["threshold"]
        pipeline_threshold_changed = result.pipeline.threshold != recorded_threshold
        result_threshold_changed = result.threshold != recorded_threshold
        if pipeline_threshold_changed or result_threshold_changed:
            raise ValueError("Recorded threshold policy changed after comparison")
        if name != comparison.winner_name and "holdout" in result.metrics:
            raise ValueError("A losing candidate has holdout metrics")
    screening = getattr(comparison, "screening_comparison", None)
    if screening is not None:
        if (
            screening is comparison
            or getattr(screening, "screening_comparison", None) is not None
        ):
            raise ValueError("Expected a single earlier screening comparison")
        validate_comparison_contract(screening, partitions, data_scope)
        if screening.metadata["comparison_id"] != comparison.metadata.get(
            "screening_comparison_id"
        ):
            raise ValueError("Detailed study identifies a different screening artifact")
        if any("holdout" in result.metrics for result in screening.results.values()):
            raise ValueError("Screening candidates must remain validation-only")
    return comparison.winner


def evaluate_selected_comparison(comparison, partitions):
    """Assess the frozen winner, preserving its object, artifact ID and threshold.

    Repeated notebook execution returns the cached assessment; it never refits or
    generates another score for tuning. A new comparison is a new experiment and
    cannot magically make a previously inspected holdout unseen again.
    """
    winner = validate_comparison_contract(comparison, partitions)
    if winner.pipeline.threshold != winner.threshold:
        raise ValueError("Selected threshold changed after comparison")
    training_ids_fingerprint = frame_fingerprint(partitions.train[["TransactionID"]])
    if winner.pipeline.training_ids_hash_ != training_ids_fingerprint:
        raise ValueError("Selected model was not fitted on the declared training rows")
    current_validation_metrics = evaluate_pipeline(
        winner.pipeline, partitions.validation
    )
    if current_validation_metrics != winner.metrics["validation"]:
        raise ValueError("Selected model validation metrics changed after comparison")
    if "holdout" not in winner.metrics:
        winner.metrics["holdout"] = evaluate_pipeline(
            winner.pipeline, partitions.holdout
        )
        winner.metadata["assessment_policy"] = (
            "frozen winner only; no post-assessment refit"
        )
        comparison.metadata["holdout_evaluated"] = True
    winner.metadata["comparison_summary"] = _summary(comparison)
    return winner


def save_comparison(comparison, output_dir=None):
    """Persist fitted candidate objects and a readable, checksum-linked report.

    Local joblib holds raw validation inputs used for equivalence checks. MLflow
    receives only the report and model, never these raw row copies. Every save
    creates a fresh folder and preserves historical experiments.
    """
    if output_dir:
        root = Path(output_dir)
    else:
        root = PROJECT_ROOT / "models" / "local" / "comparisons"
    folder = (
        root
        / f"comparison_{comparison.metadata['comparison_id']}_{uuid.uuid4().hex[:8]}"
    )
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / "comparison.joblib"
    with path.open("xb") as stream:
        joblib.dump(comparison, stream)
    record = {
        **_summary(comparison),
        "comparison_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    report_path = folder / "comparison.json"
    report_path.write_text(
        json.dumps(record, indent=2, default=_json_default), encoding="utf-8"
    )
    comparison.table.to_csv(folder / "comparison.csv", index=False)
    return folder


def load_comparison(folder):
    """Load a trusted local comparison and verify its accompanying report.

    The checksum detects changes to the saved file. It does not make an
    untrusted pickle safe: joblib files must come from this local workflow.
    """
    folder = Path(folder)
    record = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
    path = folder / "comparison.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["comparison_sha256"]:
        raise ValueError("Comparison handoff checksum mismatch")
    comparison = joblib.load(path)
    if not isinstance(comparison, ComparisonResult):
        raise ValueError("Expected a complete ComparisonResult handoff")
    winner_changed = comparison.winner_name != record["winner_name"]
    artifact_changed = (
        comparison.winner.metadata["artifact_id"] != record["winner_artifact_id"]
    )
    criterion_changed = (
        comparison.metadata["selection_criterion"] != record["selection_criterion"]
    )
    if winner_changed or artifact_changed or criterion_changed:
        raise ValueError("Comparison handoff report differs from fitted objects")
    return comparison
