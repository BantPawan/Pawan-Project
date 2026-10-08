"""Train and save complete fraud models that accept joined raw transactions.

This module does four things:

1. Build a classifier with reproducible, affordable settings.
2. Fit feature engineering, selection and preprocessing on training rows only.
3. Pick a decision threshold from validation rows and evaluate fixed models.
4. Save the fitted pipeline along with the evidence that identifies its training data.

The saved pipeline carries its own transformations, so prediction uses the
same encodings, medians and feature columns that were learned during training.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .data import PROJECT_ROOT, validate_transactions
from .features import FeatureEngineering, RawFeatureScreen
from .feature_selection import DEFAULT_CONSENSUS_COMPONENTS, FeatureSelection
from .preprocessing import Preprocessor
from .utils import calculate_metrics, frame_fingerprint, column_view


def parameter_record(value):
    """Make model parameters JSON-safe and comparable.

    Some boosting libraries use NaN as a sentinel. NaN does not equal itself,
    so we store it as a string. VotingClassifier holds estimator objects, so we
    store their class and constructor settings rather than an object repr.
    """
    if isinstance(value, BaseEstimator):
        return {
            "estimator_class": f"{type(value).__module__}.{type(value).__name__}",
            "parameters": parameter_record(value.get_params(deep=False)),
        }
    if isinstance(value, dict):
        return {str(key): parameter_record(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [parameter_record(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return f"nonfinite:{value}"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "Unsupported parameter type for reproducible provenance: "
        f"{type(value).__name__}"
    )


def build_classifier(model, seed=42, model_params=None, labels=None):
    """Build a classifier with affordable, reproducible defaults.

    XGBoost's class weight is derived from the labels passed in this call, so
    it is computed inside CV as well and never uses a validation or holdout
    class ratio. Any override supplied through ``model_params`` is applied and
    recorded in the fitted classifier.
    """
    if model == "logistic":
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        )

    elif model == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=80,
            max_depth=10,
            class_weight="balanced",
            n_jobs=1,
            random_state=seed,
        )

    elif model == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError(
                "Install requirements-local.txt to compare XGBoost"
            ) from error

        if labels is None:
            ratio = 1.0
        else:
            training_labels = np.asarray(labels)
            negative_count = np.sum(training_labels == 0)
            positive_count = np.sum(training_labels == 1)
            ratio = float(negative_count / positive_count)

        classifier = XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=ratio,
            n_jobs=1,
            random_state=seed,
        )

    elif model == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as error:
            raise ImportError(
                "Install requirements-local.txt to compare LightGBM"
            ) from error

        classifier = LGBMClassifier(
            n_estimators=80,
            num_leaves=15,
            learning_rate=0.08,
            min_child_samples=10,
            class_weight="balanced",
            verbosity=-1,
            n_jobs=1,
            random_state=seed,
            device_type="cpu",
        )

    elif model == "ensemble":
        settings = dict(model_params or {})
        members = settings.pop("members", None)

        if settings or not isinstance(members, (tuple, list)) or len(members) < 2:
            raise ValueError(
                "An equal-weight ensemble needs at least two named members, "
                "and 'members' is its only supported parameter"
            )

        estimators = []
        names = set()

        for member in members:
            name = member.get("name")
            family = member.get("model")

            if not isinstance(name, str) or not name or "__" in name or name in names:
                raise ValueError(
                    "Ensemble member names must be unique and nonempty"
                )
            if family not in ("logistic", "random_forest", "xgboost", "lightgbm"):
                raise ValueError(
                    "Ensemble members must be individual classifier families"
                )

            names.add(name)
            estimators.append(
                (name, build_classifier(family, seed, member.get("model_params"), labels))
            )

        # Weights are fixed before validation. Every member learns the same
        # training-fitted representation inside the full pipeline.
        return VotingClassifier(
            estimators=estimators,
            voting="soft",
            weights=None,
            n_jobs=1,
        )

    else:
        raise ValueError(
            "Supported models: logistic, random_forest, xgboost, lightgbm, ensemble"
        )

    overrides = dict(model_params or {})
    unknown = set(overrides) - set(classifier.get_params())
    if unknown:
        raise ValueError(f"Unknown {model} model parameters: {sorted(unknown)}")

    return classifier.set_params(**overrides)


class FraudPipeline(ClassifierMixin, BaseEstimator):
    """Fit raw transaction rows and return scores or thresholded fraud labels.

    Required columns are TransactionID, TransactionDT and TransactionAmt.
    Optional fit-time fields use the fitted missing-value policy. TransactionID
    and the target never reach the classifier.

    With automatic encoding, logistic regression receives one-hot categories
    and scaled continuous features, while tree models receive category codes
    and unscaled values. All preprocessing stays inside the sklearn pipeline.
    """

    def __init__(
        self,
        model="logistic",
        seed=42,
        n_clusters=0,
        threshold=0.5,
        encoding="auto",
        selection_method="basic",
        selection_max_features=None,
        feature_groups="all",
        model_params=None,
        raw_screen_strategy="all",
        top_v_features=None,
        preserve_features=(),
        selection_consensus_components=DEFAULT_CONSENSUS_COMPONENTS,
        selection_device="cpu",
    ):
        """Store configuration. Learned state is created only in ``fit``."""
        self.model = model
        self.seed = seed
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.encoding = encoding
        self.selection_method = selection_method
        self.selection_max_features = selection_max_features
        self.feature_groups = feature_groups
        self.model_params = model_params
        self.raw_screen_strategy = raw_screen_strategy
        self.top_v_features = top_v_features
        self.preserve_features = preserve_features
        self.selection_consensus_components = selection_consensus_components
        self.selection_device = selection_device

    def __setstate__(self, state):
        """Keep earlier saved models loadable and cloneable.

        Older pipelines predate the explicit raw-screening settings. Supplying
        the inert defaults here lets them stay cloneable without touching the
        already fitted sklearn steps.
        """
        super().__setstate__(state)

        defaults = {
            "raw_screen_strategy": "all",
            "top_v_features": None,
            "preserve_features": (),
            "selection_consensus_components": DEFAULT_CONSENSUS_COMPONENTS,
            "selection_device": "cpu",
        }

        for name, value in defaults.items():
            if not hasattr(self, name):
                setattr(self, name, value)

    def fit(self, X, y):
        """Learn every transformation and the classifier from these rows only.

        Sorting here protects callers who pass shuffled training rows. The row
        fingerprint describes the caller's original input order, so the saved
        record can be checked later against that exact frame.
        """
        validate_transactions(X.drop(columns="isFraud", errors="ignore"), labeled=False)

        if isinstance(y, pd.Series) and not X.index.equals(y.index):
            raise ValueError(
                "Training label index must align with feature row index"
            )

        labels = np.asarray(y)
        if len(labels) != len(X) or set(np.unique(labels)) != {0, 1}:
            raise ValueError(
                "Training requires aligned binary labels with both classes"
            )

        if not 0 <= self.threshold <= 1:
            raise ValueError("Decision threshold must be in [0, 1]")

        classifier = build_classifier(
            self.model, self.seed, self.model_params, labels
        )

        if self.encoding not in ("auto", "legacy", "onehot"):
            raise ValueError("encoding must be auto, legacy or onehot")

        if self.encoding == "auto":
            self.encoding_ = (
                "onehot" if self.model in ("logistic", "ensemble") else "legacy"
            )
        else:
            self.encoding_ = self.encoding

        self.pipeline = Pipeline(
            [
                (
                    "raw_screen",
                    RawFeatureScreen(
                        raw_screen_strategy=self.raw_screen_strategy,
                        top_v_features=self.top_v_features,
                        preserve_features=self.preserve_features,
                        seed=self.seed,
                    ),
                ),
                (
                    "features",
                    FeatureEngineering(
                        random_state=self.seed,
                        n_clusters=self.n_clusters,
                        encoding=self.encoding_,
                        feature_groups=self.feature_groups,
                    ),
                ),
                (
                    "selection",
                    FeatureSelection(
                        method=self.selection_method,
                        max_features=self.selection_max_features,
                        random_state=self.seed,
                        preserve_features=self.preserve_features,
                        consensus_components=self.selection_consensus_components,
                        device=self.selection_device,
                    ),
                ),
                (
                    "preprocessing",
                    Preprocessor(
                        scale=self.model in ("logistic", "ensemble"),
                        scale_categorical=False,
                    ),
                ),
                ("classifier", classifier),
            ]
        )

        # Sort inside every fit so inner selector windows and relative-time
        # features always receive chronological rows.
        order = np.argsort(X.TransactionDT.to_numpy(), kind="stable")
        raw = column_view(X, [name for name in X if name != "isFraud"])
        ordered_raw = raw if np.array_equal(order, np.arange(len(X))) else raw.iloc[order]
        ordered_labels = labels[order]

        self.pipeline.fit(ordered_raw, ordered_labels)

        self.classes_ = np.array([0, 1])
        self.training_rows_ = len(X)
        self.training_ids_hash_ = frame_fingerprint(X[["TransactionID"]])
        return self

    @staticmethod
    def _validated_inference_rows(raw):
        """Return unlabeled raw rows after enforcing the transaction contract."""
        if not isinstance(raw, pd.DataFrame):
            raise TypeError("FraudPipeline expects raw rows as a pandas DataFrame.")

        unlabeled = raw.drop(columns="isFraud", errors="ignore")
        validate_transactions(unlabeled, labeled=False)
        return unlabeled

    def transform(self, raw):
        """Apply fitted transformations after validating the raw input contract."""
        check_is_fitted(self, "pipeline")
        validated = self._validated_inference_rows(raw)
        return self.pipeline[:-1].transform(validated)

    def predict_proba(self, raw):
        """Return classifier scores after validating raw inference rows."""
        check_is_fitted(self, "pipeline")
        validated = self._validated_inference_rows(raw)
        return self.pipeline.predict_proba(validated)

    def predict(self, raw):
        """Apply the saved threshold to the fraud class score."""
        return (self.predict_proba(raw)[:, 1] >= self.threshold).astype("int64")


@dataclass
class TrainingResult:
    """One fitted artifact together with its threshold, metrics and lineage.

    Validation inputs are kept locally for persistence checks. They are not
    exported as report artifacts.
    """

    pipeline: FraudPipeline
    metrics: dict
    threshold: float
    metadata: dict
    validation_inputs: pd.DataFrame


def validate_partitions(partitions):
    """Protect assessment rows even if callers build Partitions by hand."""
    seen = set()
    previous_max = None

    for name in ("train", "validation", "holdout"):
        frame = getattr(partitions, name)
        validate_transactions(frame, labeled=True)

        if set(frame.isFraud.unique()) != {0, 1}:
            raise ValueError(f"{name} must contain both classes")

        ids = set(frame.TransactionID)
        if seen & ids:
            raise ValueError("Partitions contain overlapping transaction IDs")

        if previous_max is not None and previous_max >= frame.TransactionDT.min():
            raise ValueError("Partitions have overlapping boundary timestamps")

        seen.update(ids)
        previous_max = frame.TransactionDT.max()


def choose_threshold(y_validation, probabilities):
    """Maximize validation F1, preferring a higher threshold on ties.

    Treating precision and recall as equally important is a learning default,
    not a bank cost policy.
    """
    precision, recall, thresholds = precision_recall_curve(
        y_validation, probabilities
    )

    # The last precision/recall point has no matching threshold.
    denominator = np.maximum(
        precision[:-1] + recall[:-1], np.finfo(float).eps
    )
    scores = 2 * precision[:-1] * recall[:-1] / denominator

    best = np.flatnonzero(np.isclose(scores, scores.max()))[-1]
    return float(thresholds[best])


def evaluate_pipeline(pipeline, labeled_rows):
    """Measure the current model and threshold without fitting either one."""
    raw = labeled_rows.drop(columns="isFraud")
    predicted_labels = pipeline.predict(raw)
    fraud_scores = pipeline.predict_proba(raw)[:, 1]
    return calculate_metrics(labeled_rows.isFraud, predicted_labels, fraud_scores)


def actual_split_report(partitions):
    """Refresh counts, fingerprints and boundaries from the actual frames."""
    report = dict(partitions.report)
    report.pop("labeled_pool_fingerprint", None)

    frames = {
        name: getattr(partitions, name)
        for name in ("train", "validation", "holdout")
    }

    partition_reports = {}
    for name, frame in frames.items():
        class_counts = {
            str(label): int(frame.isFraud.eq(label).sum()) for label in (0, 1)
        }
        partition_reports[name] = {
            "rows": len(frame),
            "fingerprint": frame_fingerprint(frame),
            "elapsed_seconds_min": float(frame.TransactionDT.min()),
            "elapsed_seconds_max": float(frame.TransactionDT.max()),
            "class_counts": class_counts,
        }

    report.update(
        labeled_pool_rows=sum(len(frame) for frame in frames.values()),
        ordered_labeled_pool_fingerprint=frame_fingerprint(
            pd.concat(frames.values())
        ),
        partitions=partition_reports,
        ids_disjoint=True,
        boundary_timestamps_disjoint=True,
    )

    report["actual_row_fractions"] = {
        name: len(frame) / report["labeled_pool_rows"]
        for name, frame in frames.items()
    }
    return report


def train_candidate(
    partitions,
    seed=42,
    model="logistic",
    n_clusters=0,
    evaluate_holdout=True,
    encoding="auto",
    selection_method="basic",
    selection_max_features=None,
    feature_groups="all",
    model_params=None,
    raw_screen_strategy="all",
    top_v_features=None,
    preserve_features=(),
    selection_consensus_components=DEFAULT_CONSENSUS_COMPONENTS,
    selection_device="cpu",
):
    """Fit on training rows and choose a threshold from validation scores.

    A single-model run can assess holdout immediately. Model comparison passes
    ``evaluate_holdout=False`` here so that losing candidates never see
    holdout. ``compare_candidates`` then assesses only the frozen winner.
    """
    validate_partitions(partitions)

    raw_train = column_view(partitions.train, [c for c in partitions.train if c != "isFraud"])
    raw_validation = column_view(partitions.validation, [c for c in partitions.validation if c != "isFraud"])

    pipeline = FraudPipeline(
        model=model,
        seed=seed,
        n_clusters=n_clusters,
        encoding=encoding,
        selection_method=selection_method,
        selection_max_features=selection_max_features,
        feature_groups=feature_groups,
        model_params=model_params,
        raw_screen_strategy=raw_screen_strategy,
        top_v_features=top_v_features,
        preserve_features=preserve_features,
        selection_consensus_components=selection_consensus_components,
        selection_device=selection_device,
    )

    pipeline.fit(raw_train, partitions.train.isFraud)

    validation_scores = pipeline.predict_proba(raw_validation)[:, 1]
    pipeline.threshold = choose_threshold(
        partitions.validation.isFraud, validation_scores
    )

    metrics = {"validation": evaluate_pipeline(pipeline, partitions.validation)}
    if evaluate_holdout:
        metrics["holdout"] = evaluate_pipeline(pipeline, partitions.holdout)

    metadata = {
        "artifact_id": uuid.uuid4().hex,
        "artifact_role": "evaluated_training_candidate",
        "fit_partitions": ["train"],
        "seed": seed,
        "model": model,
        "n_clusters": n_clusters,
        "threshold_objective": "maximize validation F1",
        "encoding": pipeline.encoding_,
        "encoding_policy": encoding,
        "selection_method": selection_method,
        "selection_max_features": selection_max_features,
        "feature_groups": feature_groups,
        "raw_screen_strategy": raw_screen_strategy,
        "top_v_features": top_v_features,
        "preserve_features": tuple(preserve_features),
        "selection_consensus_components": tuple(selection_consensus_components),
        "selection_device": selection_device,
        "model_params": deepcopy(model_params or {}),
        "classifier_parameters": parameter_record(
            pipeline.pipeline.named_steps["classifier"].get_params()
        ),
        "threshold": pipeline.threshold,
        "training_rows": len(partitions.train),
        "training_ids_hash": pipeline.training_ids_hash_,
        "split_report": actual_split_report(partitions),
        "partition_fingerprints": {
            name: frame_fingerprint(getattr(partitions, name))
            for name in ("train", "validation", "holdout")
        },
        "raw_input_columns": list(raw_train.columns),
        "selected_features": list(
            pipeline.pipeline.named_steps["selection"].selected_features_
        ),
    }

    classifier = pipeline.pipeline.named_steps["classifier"]
    if isinstance(classifier, VotingClassifier):
        metadata["ensemble_policy"] = (
            "fixed equal-weight mean; no learned weights"
        )
        metadata["ensemble_member_parameters"] = {
            name: parameter_record(estimator.get_params())
            for name, estimator in classifier.named_estimators_.items()
        }

    metadata["metric_artifact_id"] = metadata["artifact_id"]

    return TrainingResult(
        pipeline, metrics, pipeline.threshold, metadata, raw_validation
    )


def refit_candidate(result, partitions):
    """Optional train+validation refit, assessed only on holdout.

    The validation-selected policy is kept. Validation metrics belong to the
    earlier candidate and are deliberately absent from the refit record.
    """
    validate_partitions(partitions)

    current = {
        name: frame_fingerprint(getattr(partitions, name))
        for name in ("train", "validation", "holdout")
    }
    if current != result.metadata["partition_fingerprints"]:
        raise ValueError(
            "Refit partitions differ from the parent candidate data lineage"
        )

    development = pd.concat(
        [partitions.train, partitions.validation], ignore_index=True
    )

    pipeline = clone(result.pipeline)
    pipeline.fit(development.drop(columns="isFraud"), development.isFraud)

    metadata = dict(result.metadata)
    metadata.update(
        artifact_id=uuid.uuid4().hex,
        parent_artifact_id=result.metadata["artifact_id"],
        artifact_role="development_refit",
        fit_partitions=["train", "validation"],
        training_rows=len(development),
        training_ids_hash=pipeline.training_ids_hash_,
    )
    metadata["selected_features"] = list(
        pipeline.pipeline.named_steps["selection"].selected_features_
    )
    metadata["classifier_parameters"] = parameter_record(
        pipeline.pipeline.named_steps["classifier"].get_params()
    )

    classifier = pipeline.pipeline.named_steps["classifier"]
    if isinstance(classifier, VotingClassifier):
        metadata["ensemble_member_parameters"] = {
            name: parameter_record(estimator.get_params())
            for name, estimator in classifier.named_estimators_.items()
        }

    metadata["split_report"] = actual_split_report(partitions)
    metadata["metric_artifact_id"] = metadata["artifact_id"]

    holdout_metrics = evaluate_pipeline(pipeline, partitions.holdout)

    return TrainingResult(
        pipeline,
        {"holdout": holdout_metrics},
        pipeline.threshold,
        metadata,
        result.validation_inputs,
    )


def save_pipeline(pipeline, path):
    """Save the complete fitted state without overwriting earlier artifacts."""
    check_is_fitted(pipeline, "pipeline")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("xb") as stream:
        joblib.dump(pipeline, stream)

    return path


def load_pipeline(path):
    """Load a trusted local artifact. Pickle files execute code, so trust matters."""
    pipeline = joblib.load(path)

    if not isinstance(pipeline, FraudPipeline) or not hasattr(pipeline, "pipeline"):
        raise ValueError(
            "Expected a complete fitted FraudPipeline; historical classifiers differ"
        )

    return pipeline


def save_results(result, output_dir=None):
    """Save the evaluated candidate as-is. Never implicitly refit."""
    root = Path(output_dir) if output_dir else PROJECT_ROOT / "models" / "local"
    folder = root / f"candidate_{result.metadata['artifact_id']}"
    folder.mkdir(parents=True, exist_ok=False)

    path = save_pipeline(result.pipeline, folder / "pipeline.joblib")

    record = {
        **result.metadata,
        "metrics": result.metrics,
        "pipeline_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

    (folder / "metadata.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return path


def generated_fixture(n_rows=240, seed=42):
    """Integration fixture. Never replaces raw files or stands in as real evidence."""
    if n_rows < 30:
        raise ValueError(
            "Use at least 30 fixture rows for chronological class coverage"
        )

    rng = np.random.default_rng(seed)

    amounts = rng.lognormal(3.5, 1.0, n_rows)
    products = np.array(["W", "C", "H", "R"])[np.arange(n_rows) % 4]
    labels = (
        (np.arange(n_rows) % 7 == 0) | ((amounts > 90) & (products == "C"))
    ).astype(int)

    frame = pd.DataFrame(
        {
            "TransactionID": np.arange(100000, 100000 + n_rows),
            "TransactionDT": np.arange(n_rows, dtype=float) * 3600,
            "TransactionAmt": amounts,
            "ProductCD": products,
            "card1": 1000 + np.arange(n_rows) % 17,
            "P_emaildomain": np.where(
                np.arange(n_rows) % 3 == 0, None, "example.test"
            ),
            "C1": rng.integers(0, 9, n_rows).astype(float),
            "V1": rng.normal(size=n_rows),
            "id_30": np.where(
                np.arange(n_rows) % 4 == 0, "Windows 10", None
            ),
            "isFraud": labels,
        }
    )

    frame.attrs["data_scope"] = "generated_fixture"
    return frame


class ModelSelection:
    """Expose the original model-study stages through the shared safe workflow.

    Candidate initialization, two-phase evaluation and final persistence stay
    visible as methods. They accept raw chronological partitions; previously
    encoded or selected matrices cannot bypass fold-specific fitting.
    """

    train_candidate = staticmethod(train_candidate)
    save_results = staticmethod(save_results)
    refit_candidate = staticmethod(refit_candidate)

    def __init__(self, seed=42, n_splits=3, tune=True):
        """Configure reproducibility and the bounded comparison workload."""
        self.seed = seed
        self.n_splits = n_splits
        self.tune = tune

    def initialize_models(self, smoke=False, gpu=False):
        """Declare the four original classifier families before scoring them."""
        from .experiments import default_candidates

        return default_candidates(smoke=smoke, gpu=gpu)

    def run_two_phase_evaluation(self, partitions, candidates=None, **study_options):
        """Screen, promote and tune complete pipelines using past training folds.

        The legacy default of three screening folds leads to five detailed
        folds. Callers can declare smaller smoke budgets explicitly. Holdout
        is still unassessed when this method returns the comparison.
        """
        from .experiments import staged_model_study

        settings = {
            "seed": self.seed,
            "screening_folds": self.n_splits,
            "detail_folds": self.n_splits + 2,
            "tune": self.tune,
            **study_options,
        }

        self.comparison_ = staged_model_study(
            partitions, candidates=candidates, **settings
        )
        return self.comparison_

    def run(self, partitions=None, candidates=None, raw_dir=None, **study_options):
        """Run the declared study, assess its frozen winner and save it.

        Supplying partitions keeps the preparation handoff visible. Otherwise
        preparation loads genuine raw files and records whether a source
        manifest establishes a bounded repository sample.
        """
        from .data_preparation import DataPreparation
        from .experiments import evaluate_selected_comparison, save_comparison

        if partitions is None:
            preparation = DataPreparation(raw_dir=raw_dir)
            partitions, _ = preparation.run()

            scope = "real_sample" if preparation.source_manifest else "real_full"
            study_options.setdefault("data_scope", scope)

        elif raw_dir is not None:
            raise ValueError(
                "Supply either prepared partitions or raw_dir, not both"
            )

        self.comparison_ = self.run_two_phase_evaluation(
            partitions,
            candidates=candidates,
            **study_options,
        )

        result = evaluate_selected_comparison(self.comparison_, partitions)
        self.comparison_folder_ = save_comparison(self.comparison_)

        return result, save_results(result)
