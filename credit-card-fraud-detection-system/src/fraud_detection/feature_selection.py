"""Select encoded features using training data and freeze that schema.

The default selector removes unusable and strongly correlated columns.
Optional supervised methods rank the remaining features using training
labels. The module also provides descriptive comparison and stability
helpers so notebook readers can see why a feature was kept or dropped.

Fit these selectors inside the full raw-row Pipeline for every
evaluation fold. A ranking computed on the whole training pool is fine
for exploration, but it is not an independent estimate of model quality.
"""

import hashlib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from .utils import column_view, numeric_feature_view


DEFAULT_CONSENSUS_COMPONENTS = ("target_correlation", "random_forest", "l1")

ALL_CONSENSUS_COMPONENTS = (
    "target_correlation",
    "random_forest",
    "lightgbm",
    "gradient_boosting",
    "permutation",
    "l1",
    "rfe",
    "shap",
)


# ---------------------------------------------------------------------
# Consensus ranking
# ---------------------------------------------------------------------

def _estimator_record(value):
    """Record estimators and their settings without unstable object reprs."""
    if isinstance(value, BaseEstimator):
        return {
            "estimator_class": f"{type(value).__module__}.{type(value).__name__}",
            "parameters": _estimator_record(value.get_params(deep=False)),
        }
    if isinstance(value, dict):
        return {str(k): _estimator_record(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_estimator_record(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return f"nonfinite:{value}"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot record ranking parameter {type(value).__name__}.")


def rank_consensus(rankings):
    """Combine within-method percentile ranks instead of raw units.

    ``rankings`` maps method names to DataFrames with ``feature`` and
    ``importance`` columns. Higher importance is better within a method.
    Tied values share a rank. Features missing from a method get zero for
    that method and an explicit coverage count. Multiplying one method's
    scores by a positive constant changes no result.
    """
    if not rankings:
        raise ValueError("Consensus requires at least one method ranking.")

    normalized = {}
    available = {}

    for method, ranking in rankings.items():
        if not isinstance(ranking, pd.DataFrame) or not {
            "feature",
            "importance",
        }.issubset(ranking):
            raise ValueError("Each consensus input needs feature and importance columns.")

        if ranking.feature.duplicated().any() or ranking.feature.isna().any():
            raise ValueError("Consensus rankings must contain unique feature names.")

        scores = pd.Series(
            pd.to_numeric(ranking.importance, errors="raise").to_numpy(),
            index=ranking.feature,
        )
        if scores.empty:
            raise ValueError("Consensus cannot combine an empty ranking.")

        scores = scores.replace([np.inf, -np.inf], np.nan)
        ranks = scores.rank(method="average", ascending=False, na_option="bottom")

        normalized[method] = 1.0 - (ranks - 1.0) / max(len(scores) - 1, 1)
        normalized[method].loc[scores.isna()] = 0.0
        available[method] = set(scores.index)

    table = pd.DataFrame(normalized).fillna(0.0)
    table.index.name = "feature"
    table["importance"] = table[list(normalized)].mean(axis=1)
    table["method_coverage"] = [
        sum(name in names for names in available.values())
        for name in table.index
    ]
    table = table.reset_index().sort_values(
        ["importance", "feature"], ascending=[False, True], kind="stable"
    )
    table["rank"] = table.importance.rank(method="min", ascending=False).astype(int)
    return table.reset_index(drop=True)


# ---------------------------------------------------------------------
# Main selector
# ---------------------------------------------------------------------

class FeatureSelection(TransformerMixin, BaseEstimator):
    """Filter numeric features and optionally rank them with training labels.

    Correlations are measured before imputation. When two features are
    correlated, the earlier column in the fitted schema is kept. Apply the
    frozen selection to later partitions regardless of their missingness,
    correlations or labels.

    ``basic`` keeps the previous unsupervised behaviour. Other methods
    rank the remaining columns and keep at most ``max_features`` (half by
    default). All scoring models and their preprocessing are fitted only
    inside this ``fit`` call. Permutation importance evaluates its scoring
    model on a later *inner* training window; the caller's outer validation
    is never passed to it. Run this transformer inside the full Pipeline
    for every evaluation fold.

    ``consensus_components`` lists the scoring methods to combine. The
    historical correlation/RF/L1 tuple stays the default. Pass
    ``ALL_CONSENSUS_COMPONENTS`` for an explicit eight-method experiment.
    Every requested component must run on the same filtered training
    schema.

    ``device='gpu'`` runs LightGBM rankings on its GPU backend. Other
    ranking methods keep their existing execution backend.
    """

    METHODS = (
        "basic",
        "target_correlation",
        "random_forest",
        "lightgbm",
        "gradient_boosting",
        "permutation",
        "l1",
        "rfe",
        "shap",
        "consensus",
    )

    ALIASES = {"permutation_importance": "permutation", "l1_logistic": "l1"}

    def __init__(
        self,
        correlation_threshold=0.95,
        missing_threshold=0.8,
        method="basic",
        max_features=None,
        random_state=42,
        preserve_features=(),
        consensus_components=DEFAULT_CONSENSUS_COMPONENTS,
        device="cpu",
    ):
        self.correlation_threshold = correlation_threshold
        self.missing_threshold = missing_threshold
        self.method = method
        self.max_features = max_features
        self.random_state = random_state
        self.preserve_features = preserve_features
        self.consensus_components = consensus_components
        self.device = device

    def __setstate__(self, state):
        """Restore cloneable defaults for earlier fitted selections."""
        super().__setstate__(state)
        if not hasattr(self, "consensus_components"):
            self.consensus_components = DEFAULT_CONSENSUS_COMPONENTS
        if not hasattr(self, "device"):
            self.device = "cpu"

    # -----------------------------------------------------------------
    # Validation helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _validate(X):
        """Copy numeric predictors, excluding identifiers and target."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureSelection expects a numeric pandas DataFrame.")
        if X.columns.has_duplicates:
            raise ValueError("Feature input contains duplicate column names.")

        names = [name for name in X if name not in ("TransactionID", "isFraud")]
        if any(not pd.api.types.is_numeric_dtype(X[name]) for name in names):
            raise ValueError("FeatureSelection expects numeric encoded features.")

        return numeric_feature_view(X, names)

    @staticmethod
    def _validate_target(y, n_rows):
        """Require one aligned binary label per row and both classes."""
        labels = np.asarray(y)

        if labels.ndim != 1 or len(labels) != n_rows:
            raise ValueError(
                "Selection labels must be one-dimensional and match the training rows."
            )

        try:
            values = labels.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Supervised selection requires binary labels 0 and 1."
            ) from exc

        if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
            raise ValueError(
                "Supervised selection requires finite binary labels 0 and 1."
            )
        if len(np.unique(values)) != 2:
            raise ValueError(
                "Supervised selection requires both label classes in training."
            )

        return values.astype(int)

    # -----------------------------------------------------------------
    # Scoring pipelines
    # -----------------------------------------------------------------

    def _scoring_pipeline(self, estimator, scale=False):
        """Give each ranking estimator its own training-fitted imputer."""
        steps = [
            ("imputation", SimpleImputer(strategy="median", keep_empty_features=True))
        ]
        if scale:
            steps.append(("scaling", StandardScaler()))
        return Pipeline(steps + [("estimator", estimator)])

    def _forest(self, n_estimators=128):
        """Bounded, reproducible forest used for training rankings."""
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=1,
        )

    def _permutation_scores(self, frame, labels, times):
        """Score importance on a later window inside the supplied training set.

        The scoring estimator and imputer fit only the earlier inner window.
        Upstream feature engineering and basic filters use the outer training
        window, so this score is a descriptive diagnostic, not an independent
        validation metric.
        """
        if times is None:
            raise ValueError(
                "Permutation selection requires TransactionDT for an inner split."
            )

        times = np.asarray(times, dtype=float)
        if not np.isfinite(times).all() or (times < 0).any():
            raise ValueError(
                "Permutation selection requires finite nonnegative TransactionDT."
            )

        unique_times = np.unique(times)
        if len(unique_times) < 2:
            raise ValueError(
                "Permutation selection needs at least two distinct transaction times."
            )

        # Rows with the same time stay together, even if input was shuffled.
        boundary = min(
            max(int(np.ceil(0.8 * len(unique_times))), 1), len(unique_times) - 1
        )
        train_positions = np.flatnonzero(times < unique_times[boundary])
        evaluation_positions = np.flatnonzero(times >= unique_times[boundary])

        train_positions = train_positions[
            np.argsort(times[train_positions], kind="stable")
        ]
        evaluation_positions = evaluation_positions[
            np.argsort(times[evaluation_positions], kind="stable")
        ]

        if any(
            len(np.unique(labels[pos])) != 2
            for pos in (train_positions, evaluation_positions)
        ):
            raise ValueError(
                "Both classes are required in each inner permutation window."
            )

        self.ranking_estimator_ = self._scoring_pipeline(self._forest())
        self.ranking_estimator_.fit(frame.iloc[train_positions], labels[train_positions])

        evaluation = frame.iloc[evaluation_positions]
        evaluation_labels = labels[evaluation_positions]

        result = permutation_importance(
            self.ranking_estimator_,
            evaluation,
            labels[evaluation_positions],
            scoring="average_precision",
            n_repeats=3,
            random_state=self.random_state,
            n_jobs=1,
        )

        self.inner_split_ = {
            "source": "selector_training_rows_only",
            "estimator_fit_rows": len(train_positions),
            "evaluation_rows": len(evaluation_positions),
            "train_positions": train_positions.tolist(),
            "evaluation_positions": evaluation_positions.tolist(),
            "train_time_min": float(times[train_positions].min()),
            "train_time_max": float(times[train_positions].max()),
            "evaluation_time_min": float(times[evaluation_positions].min()),
            "evaluation_time_max": float(times[evaluation_positions].max()),
            "train_class_counts": {
                str(k): int((labels[train_positions] == k).sum()) for k in (0, 1)
            },
            "evaluation_class_counts": {
                str(k): int((labels[evaluation_positions] == k).sum()) for k in (0, 1)
            },
            "average_precision": float(
                average_precision_score(
                    evaluation_labels,
                    self.ranking_estimator_.predict_proba(evaluation)[:, 1],
                )
            ),
            "scope": "inner estimator and imputer; outer features/basic filters use the outer training window",
        }

        return result.importances_mean

    # -----------------------------------------------------------------
    # Ranking methods
    # -----------------------------------------------------------------

    def _rank(self, frame, labels, limit, times):
        """Return method-specific scores for already filtered training features.

        Scores from different methods have different units. They rank
        features within one experiment and should not be compared as
        equal quantities across methods.
        """
        method = self.method_

        if method == "consensus":
            return self._rank_consensus(frame, labels, limit, times)

        if method == "target_correlation":
            self.importance_kind_ = "absolute_pearson_target_correlation"
            target = pd.Series(labels, index=frame.index)
            return frame.corrwith(target).abs().fillna(0).to_numpy()

        if method == "permutation":
            self.importance_kind_ = "inner_average_precision_decrease"
            return self._permutation_scores(frame, labels, times)

        if method in ("l1", "rfe"):
            return self._rank_l1_or_rfe(method, frame, labels, limit)

        if method == "lightgbm":
            return self._rank_lightgbm(frame, labels)

        if method == "gradient_boosting":
            return self._rank_gradient_boosting(frame, labels)

        if method == "shap":
            return self._rank_shap(frame, labels)

        # Fallback covers random_forest.
        return self._rank_tree(frame, labels, n_estimators=128)

    def _rank_consensus(self, frame, labels, limit, times):
        """Rank features with all declared components on the same schema."""
        component_rankings = {}
        self.consensus_estimators_ = {}
        self.consensus_component_reports_ = {}

        for component in self.consensus_components_:
            scorer = FeatureSelection(
                method=component, random_state=self.random_state, device=self.device
            )
            scorer.method_ = component
            status = "fitted"

            try:
                scores = scorer._rank(frame, labels, limit, times)
            except ImportError as exc:
                raise ImportError(
                    f"Consensus component {component!r} is unavailable: {exc}"
                ) from exc
            except ValueError as exc:
                if component == "l1" and "no nonzero coefficients" in str(exc):
                    # Zero evidence, not a missing component. Preserves
                    # previous default consensus policy.
                    scores = np.zeros(frame.shape[1])
                    status = "fitted_zero_coefficients"
                else:
                    raise

            scores = np.asarray(scores, dtype=float)
            if scores.shape != (frame.shape[1],) or not np.isfinite(scores).all():
                raise ValueError(
                    f"Consensus component {component!r} must score every feature."
                )

            component_rankings[component] = pd.DataFrame(
                {"feature": frame.columns, "importance": scores}
            )

            if hasattr(scorer, "ranking_estimator_"):
                self.consensus_estimators_[component] = scorer.ranking_estimator_

            self.consensus_component_reports_[component] = {
                "method": component,
                "device": scorer.device,
                "status": status,
                "importance_kind": scorer.importance_kind_,
                "input_features": frame.columns.tolist(),
                "input_rows": len(frame),
                "ranking_pipeline": (
                    _estimator_record(scorer.ranking_estimator_)
                    if hasattr(scorer, "ranking_estimator_")
                    else None
                ),
            }

            for detail in ("inner_split_", "shap_sample_positions_"):
                if hasattr(scorer, detail):
                    self.consensus_component_reports_[component][
                        detail[:-1]
                    ] = getattr(scorer, detail)

        self.consensus_ranks_ = rank_consensus(component_rankings)
        self.consensus_component_rankings_ = component_rankings
        self.importance_kind_ = "mean_within_method_normalized_rank"

        return (
            self.consensus_ranks_
            .set_index("feature")
            .loc[frame.columns, "importance"]
            .to_numpy()
        )

    def _rank_l1_or_rfe(self, method, frame, labels, limit):
        """L1 logistic coefficients or inverse RFE rank."""
        estimator = LogisticRegression(
            penalty="l1" if method == "l1" else "l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=self.random_state,
        )

        if method == "rfe":
            if limit >= frame.shape[1]:
                raise ValueError(
                    "RFE must eliminate a feature: max_features must be below "
                    "the filtered feature count."
                )
            estimator = RFE(estimator, n_features_to_select=limit, step=0.2)

        self.ranking_estimator_ = self._scoring_pipeline(estimator, scale=True).fit(
            frame, labels
        )
        fitted = self.ranking_estimator_.named_steps["estimator"]

        if method == "rfe":
            self.importance_kind_ = "inverse_rfe_rank"
            self.rfe_support_ = fitted.support_.copy()
            return 1.0 / fitted.ranking_

        self.importance_kind_ = "absolute_standardized_l1_coefficient"
        scores = np.abs(fitted.coef_).mean(axis=0)

        if not (scores > 0).any():
            raise ValueError(
                "L1 selection found no nonzero coefficients; use a different selector."
            )

        return scores

    def _rank_lightgbm(self, frame, labels):
        """LightGBM feature importance. Optional dependency."""
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "method='lightgbm' requires the optional lightgbm dependency."
            ) from exc

        estimator = LGBMClassifier(
            n_estimators=80,
            max_depth=6,
            num_leaves=31,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=1,
            verbosity=-1,
            device_type=self.device,
        )
        return self._rank_tree(frame, labels, estimator=estimator)

    def _rank_gradient_boosting(self, frame, labels):
        """Gradient boosting feature importance."""
        estimator = GradientBoostingClassifier(
            n_estimators=80, max_depth=3, random_state=self.random_state
        )
        return self._rank_tree(frame, labels, estimator=estimator)

    def _rank_tree(self, frame, labels, estimator=None, n_estimators=128):
        """Random forest feature importance (also used by lightgbm)."""
        if estimator is None:
            estimator = self._forest(n_estimators=n_estimators)

        self.ranking_estimator_ = self._scoring_pipeline(estimator).fit(frame, labels)
        fitted = self.ranking_estimator_.named_steps["estimator"]
        self.importance_kind_ = "tree_feature_importance"

        return fitted.feature_importances_

    def _rank_shap(self, frame, labels):
        """Mean absolute SHAP values on a bounded training sample."""
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "method='shap' requires the optional shap dependency."
            ) from exc

        estimator = self._forest(n_estimators=64)
        self.ranking_estimator_ = self._scoring_pipeline(estimator).fit(frame, labels)
        fitted = self.ranking_estimator_.named_steps["estimator"]

        rng = np.random.default_rng(self.random_state)
        positions = np.sort(
            rng.choice(len(frame), size=min(512, len(frame)), replace=False)
        )
        values = self.ranking_estimator_.named_steps["imputation"].transform(
            frame.iloc[positions]
        )

        explanations = shap.TreeExplainer(fitted).shap_values(values)
        if isinstance(explanations, list):
            scores = np.abs(np.stack(explanations)).mean(axis=(0, 1))
        else:
            explanations = np.asarray(explanations)
            scores = np.abs(explanations).mean(
                axis=(0, 2) if explanations.ndim == 3 else 0
            )

        self.shap_sample_positions_ = positions.tolist()
        self.importance_kind_ = "mean_absolute_training_shap"

        return scores

    # -----------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------

    def fit(self, X, y=None):
        """Learn removals, optional rankings, and an ordered training schema.

        Basic filtering ignores labels. Supervised methods require aligned
        binary labels and rank at most ``max_features``. When no limit is
        supplied, they rank half of the filtered columns. Declared usable
        preservation requests are added after the ranked budget and may
        exceed it. Reports keep removals and fitting scope visible.
        """
        if (
            isinstance(y, pd.Series)
            and isinstance(X, pd.DataFrame)
            and not X.index.equals(y.index)
        ):
            raise ValueError("Selection label index must align with feature row index.")

        if (
            not 0 <= self.correlation_threshold <= 1
            or not 0 <= self.missing_threshold <= 1
        ):
            raise ValueError("Selection thresholds must lie between 0 and 1.")

        if self.device not in ("cpu", "gpu"):
            raise ValueError("Selection device must be cpu or gpu.")

        self.method_ = self.ALIASES.get(self.method, self.method)
        if self.method_ not in self.METHODS:
            raise ValueError(
                f"Unknown selection method {self.method!r}; choose from {self.METHODS}."
            )

        self._validate_consensus_components()
        self._validate_max_features()
        self._validate_preserve_features()

        # Drop any reports/estimators from a previous fit.
        for name in (
            "inner_split_",
            "ranking_estimator_",
            "rfe_support_",
            "shap_sample_positions_",
            "consensus_estimators_",
            "consensus_ranks_",
            "consensus_component_rankings_",
            "consensus_component_reports_",
        ):
            self.__dict__.pop(name, None)

        frame = self._validate(X)
        if frame.empty:
            raise ValueError("Cannot fit feature selection on an empty training frame.")

        self.input_columns_ = frame.columns.tolist()
        self.feature_names_in_ = np.asarray(self.input_columns_, dtype=object)
        self.n_features_in_ = len(self.input_columns_)
        self.missingness_ = frame.isna().mean().to_dict()

        self.high_missing_columns_ = [
            name for name in frame if self.missingness_[name] > self.missing_threshold
        ]
        self.constant_columns_ = [
            name for name in frame if frame[name].nunique(dropna=True) <= 1
        ]

        excluded = set(self.high_missing_columns_ + self.constant_columns_)
        candidates = [name for name in frame if name not in excluded]

        preserved = self._resolve_preserved(candidates)

        correlation_inputs = column_view(frame, candidates)
        if self.device == "gpu":
            from .gpu_correlation import pairwise_pearson
            correlations, self.correlation_report_ = pairwise_pearson(
                correlation_inputs,
                threshold=self.correlation_threshold,
                return_report=True,
            )
        else:
            correlations = correlation_inputs.corr()
            self.correlation_report_ = {"backend": "pandas", "dtype": "float64"}
        correlations = correlations.abs()
        self.selected_features_ = []
        self.correlated_columns_ = []

        for name in candidates:
            if name not in preserved and any(
                correlations.loc[name, selected] > self.correlation_threshold
                for selected in self.selected_features_
            ):
                self.correlated_columns_.append(name)
            else:
                self.selected_features_.append(name)

        if not self.selected_features_:
            raise ValueError("No usable features remain after training-only selection.")

        self.filtered_features_ = self.selected_features_.copy()

        if self.method_ == "basic":
            labels = None
            limit = len(self.filtered_features_)
        else:
            labels = self._validate_target(y, len(frame))
            limit = max(1, len(self.filtered_features_) // 2)

        if self.max_features is not None:
            limit = min(self.max_features, len(self.filtered_features_))

        self.importance_kind_ = "input_order_basic_filter"

        if labels is None:
            scores = np.full(len(self.filtered_features_), np.nan)
        else:
            times = frame["TransactionDT"] if "TransactionDT" in frame else None
            scores = self._rank(column_view(frame, self.filtered_features_), labels, limit, times)

        ranking = (
            pd.DataFrame({"feature": self.filtered_features_, "importance": scores})
            .sort_values("importance", ascending=False, kind="stable", na_position="last")
        )

        retained = ranking.head(limit)
        if self.method_ == "l1":
            retained = retained[retained.importance > 0]

        selected = set(retained.feature)
        selected.update(preserved)

        self.selected_features_ = [
            name for name in self.filtered_features_ if name in selected
        ]

        ranking["rank"] = np.arange(1, len(ranking) + 1)
        ranking["selected"] = ranking.feature.isin(self.selected_features_)
        self.importances_ = ranking.reset_index(drop=True)

        self.selection_report_ = self._build_report(
            frame, labels, limit, preserved, ranking
        )
        self.selection_report_["correlation"] = self.correlation_report_

        self.output_features_ = self.selected_features_.copy()
        return self

    def _validate_consensus_components(self):
        components = getattr(self, "consensus_components", DEFAULT_CONSENSUS_COMPONENTS)

        if not isinstance(components, (tuple, list)) or not components:
            raise ValueError("consensus_components must be a nonempty tuple or list.")

        if any(not isinstance(name, str) for name in components):
            raise ValueError("consensus_components must contain method names.")

        self.consensus_components_ = tuple(
            self.ALIASES.get(name, name) for name in components
        )

        if len(set(self.consensus_components_)) != len(self.consensus_components_):
            raise ValueError("consensus_components must contain unique methods.")

        if any(
            name not in ALL_CONSENSUS_COMPONENTS
            for name in self.consensus_components_
        ):
            raise ValueError(
                f"Consensus components must be scoring methods from "
                f"{ALL_CONSENSUS_COMPONENTS}."
            )

    def _validate_max_features(self):
        if self.max_features is None:
            return
        if (
            isinstance(self.max_features, bool)
            or not isinstance(self.max_features, (int, np.integer))
            or self.max_features < 1
        ):
            raise ValueError("max_features must be a positive integer or None.")

    def _validate_preserve_features(self):
        if not isinstance(self.preserve_features, (tuple, list)) or any(
            not isinstance(name, str) for name in self.preserve_features
        ):
            raise ValueError("preserve_features must contain feature names.")

        if any(name in ("TransactionID", "isFraud") for name in self.preserve_features):
            raise ValueError(
                "Identifiers and labels cannot be preserved as predictors."
            )

    def _resolve_preserved(self, candidates):
        """Expand nominal requests into their frozen encoded column names."""
        requested = set(self.preserve_features)

        preserved = {
            name
            for name in candidates
            if name in requested
            or any(
                name in (f"{source}_encoded", f"{source}_freq")
                or name.startswith(f"{source}_onehot_")
                for source in requested
            )
        }
        return preserved

    def _build_report(self, frame, labels, limit, preserved, ranking):
        """Assemble the descriptive selection report."""
        report_labels = labels
        if report_labels is None and self.preserve_features is not None:
            # Basic filtering still ignores y; record valid supplied labels
            # as descriptive provenance without changing behaviour.
            try:
                supplied = np.asarray(self._last_y, dtype=float) if hasattr(
                    self, "_last_y"
                ) else None
            except (TypeError, ValueError):
                supplied = None

        class_counts = {}
        if labels is not None:
            class_counts = {
                str(label): int((labels == label).sum()) for label in (0, 1)
            }

        if hasattr(self, "inner_split_"):
            ranking_fit_rows = self.inner_split_["estimator_fit_rows"]
        elif labels is not None:
            ranking_fit_rows = len(frame)
        else:
            ranking_fit_rows = 0

        report = {
            "method": self.method_,
            "device": self.device,
            "fit_rows": len(frame),
            "target_used": labels is not None,
            "input_feature_count": len(self.input_columns_),
            "filtered_feature_count": len(self.filtered_features_),
            "selected_feature_count": len(self.selected_features_),
            "class_counts": class_counts,
            "importance_kind": self.importance_kind_,
            "ranked_features": ranking.feature.tolist(),
            "selected_features": self.selected_features_.copy(),
            "high_missing_columns": self.high_missing_columns_.copy(),
            "constant_columns": self.constant_columns_.copy(),
            "correlated_columns": self.correlated_columns_.copy(),
            "requested_max_features": self.max_features,
            "filter_fit_rows": len(frame),
            "ranking_fit_rows": ranking_fit_rows,
            "requested_preserve_features": list(self.preserve_features),
            "retained_by_request": sorted(preserved),
            "budget_exceeded_for_retention": len(self.selected_features_) > limit,
            "unavailable_preserve_features": [
                source
                for source in self.preserve_features
                if not any(
                    name == source
                    or name in (f"{source}_encoded", f"{source}_freq")
                    or name.startswith(f"{source}_onehot_")
                    for name in preserved
                )
            ],
        }

        if self.method_ == "consensus":
            report["consensus_methods"] = list(self.consensus_component_rankings_)
            report["consensus_policy"] = (
                "mean normalized rank; equal scores share ranks; "
                "no averaging of raw units"
            )
            report["consensus_component_reports"] = self.consensus_component_reports_

        if "TransactionDT" in frame and np.isfinite(frame.TransactionDT).all():
            report.update(
                fit_time_min=float(frame.TransactionDT.min()),
                fit_time_max=float(frame.TransactionDT.max()),
                chronological_input=bool(
                    frame.TransactionDT.is_monotonic_increasing
                ),
            )

        if hasattr(self, "inner_split_"):
            report["inner_split"] = self.inner_split_

        if hasattr(self, "shap_sample_positions_"):
            report["shap_sample_positions"] = self.shap_sample_positions_

        return report

    # -----------------------------------------------------------------
    # Transform
    # -----------------------------------------------------------------

    def transform(self, X):
        """Copy the selected columns in fitted order without reselection."""
        check_is_fitted(self, "selected_features_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureSelection expects a numeric pandas DataFrame.")
        if X.columns.has_duplicates:
            raise ValueError("Feature input contains duplicate column names.")
        if any(not pd.api.types.is_numeric_dtype(X[name]) for name in X
               if name not in ("TransactionID", "isFraud")):
            raise ValueError("FeatureSelection expects numeric encoded features.")
        missing = [name for name in self.selected_features_ if name not in X]
        if missing:
            raise ValueError(f"Missing selected feature columns: {missing}")
        # Validate/clean only retained features during inference.
        frame = self._validate(column_view(X, self.selected_features_))
        return frame.copy()

    def get_feature_names_out(self, input_features=None):
        """Return the ordered schema chosen during the last successful fit."""
        check_is_fitted(self, "selected_features_")
        return np.asarray(self.selected_features_, dtype=object)

    # -----------------------------------------------------------------
    # Legacy helper
    # -----------------------------------------------------------------

    def analyze_feature_correlation(self, train_data, test_data=None, threshold=None):
        """Migration helper with an explicitly passed test frame.

        New code uses ``fit`` / ``transform`` inside a Pipeline. This helper
        fixes the former undefined ``test_data`` reference. A one-frame call
        returns a frame; a two-frame call returns the same fitted selection
        on both frames.
        """
        selector = FeatureSelection(
            correlation_threshold=(
                self.correlation_threshold if threshold is None else threshold
            ),
            missing_threshold=self.missing_threshold,
            device=self.device,
        ).fit(train_data)

        selected_train = selector.transform(train_data)

        if test_data is None:
            return selected_train

        return selected_train, selector.transform(test_data)


# ---------------------------------------------------------------------
# Descriptive helpers
# ---------------------------------------------------------------------

def compare_selection_methods(
    X,
    y,
    methods=("basic", "random_forest", "l1", "rfe"),
    max_features=None,
    random_state=42,
    consensus_components=DEFAULT_CONSENSUS_COMPONENTS,
    device="cpu",
):
    """Describe selectors on supplied training features; not model evaluation.

    Returns ``(table, fitted_selectors)``. Missing optional dependencies are
    marked ``skipped`` and invalid experiments are marked ``failed``. Fitting
    encoders on the whole training pool is fine here. Performance evaluation
    must refit the full raw-data Pipeline inside each chronological fold.
    """
    if len(set(methods)) != len(methods):
        raise ValueError("Selection comparison methods must be unique.")

    rows = []
    selectors = {}

    for method in methods:
        selector = FeatureSelection(
            method=method,
            max_features=max_features,
            random_state=random_state,
            consensus_components=consensus_components,
            device=device,
        )
        try:
            selector.fit(X, y)
        except ImportError as exc:
            rows.append(
                {
                    "method": method,
                    "status": "skipped",
                    "selected_count": None,
                    "detail": str(exc),
                }
            )
        except ValueError as exc:
            rows.append(
                {
                    "method": method,
                    "status": "failed",
                    "selected_count": None,
                    "detail": str(exc),
                }
            )
        else:
            selectors[method] = selector
            rows.append(
                {
                    "method": method,
                    "status": "fitted",
                    "selected_count": len(selector.selected_features_),
                    "detail": selector.importance_kind_,
                }
            )

    table = pd.DataFrame(rows, columns=["method", "status", "selected_count", "detail"])
    return table, selectors


def _stability_feature_identities(features, output_columns):
    """Give refitted indicators stable meaning across different windows.

    The numeric one-hot suffix is a local category position, not its
    identity. Hash the fitted tagged category value instead. Independent
    KMeans labels can permute; compare cluster features as one declared
    family only and make that limitation explicit in the report.
    """
    identities = {name: name for name in output_columns}

    for source, outputs in getattr(features, "onehot_columns_", {}).items():
        code_to_category = {
            code: category
            for category, code in features.category_maps_[source].items()
        }
        for code, output_name in outputs.items():
            if code == 0:
                category_identity = "missing"
            elif code == -1:
                category_identity = "unknown"
            else:
                category = code_to_category[code]
                token = hashlib.sha256(category.encode("utf-8")).hexdigest()
                category_identity = f"category_sha256={token}"
            identities[output_name] = f"{source}::{category_identity}"

    for name in output_columns:
        if name == "behavioral_cluster" or name.startswith("behavioral_cluster_onehot_"):
            identities[name] = "behavioral_cluster::family"

    return identities


def feature_selection_stability(
    raw_train,
    y=None,
    method="random_forest",
    fractions=(0.6, 0.8, 1.0),
    encoding="legacy",
    max_features=None,
    random_state=42,
    *,
    feature_groups="all",
    n_clusters=0,
    raw_screen_strategy="all",
    top_v_features=None,
    preserve_features=(),
    selection_consensus_components=DEFAULT_CONSENSUS_COMPONENTS,
    selection_device="cpu",
):
    """Refit features and selection on earlier raw training windows.

    Returns frequency and pairwise Jaccard DataFrames plus per-window
    reports. Equal-time transactions stay in the same window. This
    describes stability on training data; it does not estimate holdout
    performance. Pass the same screening, engineering and retention
    options as the chosen recipe. Each window learns its own V ranking,
    encoding, cluster model and selector. Different fitted encoders can
    produce different schemas, so the frequency table also records feature
    availability. One-hot categories use semantic identities; independent
    cluster labels are compared only at family level.
    """
    from .features import FeatureEngineering, RawFeatureScreen

    if not isinstance(raw_train, pd.DataFrame) or "TransactionDT" not in raw_train:
        raise ValueError(
            "Stability analysis requires a raw DataFrame containing TransactionDT."
        )

    if y is None:
        if "isFraud" not in raw_train:
            raise ValueError("Supply binary training labels or an isFraud column.")
        y = raw_train.isFraud

    if isinstance(y, pd.Series) and not raw_train.index.equals(y.index):
        raise ValueError(
            "Stability label index must align with raw training row index."
        )

    labels = FeatureSelection._validate_target(y, len(raw_train))

    times = pd.to_numeric(raw_train.TransactionDT, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(times).all() or (times < 0).any():
        raise ValueError(
            "Stability analysis requires finite nonnegative transaction times."
        )

    fractions = tuple(fractions)
    if (
        not fractions
        or any(not np.isfinite(v) or not 0 < v <= 1 for v in fractions)
        or any(left >= right for left, right in zip(fractions, fractions[1:]))
    ):
        raise ValueError(
            "Stability fractions must be increasing values greater than 0 and at most 1."
        )

    order = np.argsort(times, kind="stable")
    ordered_times = times[order]

    reports = []
    selected_sets = {}
    available_sets = {}
    inspection_names = {}

    for fraction in fractions:
        target_rows = max(1, int(np.ceil(fraction * len(raw_train))))
        cutoff = ordered_times[target_rows - 1]
        positions = order[ordered_times <= cutoff]
        window_labels = labels[positions]

        report = {
            "fraction": float(fraction),
            "fit_rows": len(positions),
            "time_min": float(times[positions].min()),
            "time_max": float(cutoff),
            "class_counts": {
                str(k): int((window_labels == k).sum()) for k in (0, 1)
            },
            "scope": "training_only_descriptive_stability",
        }

        try:
            FeatureSelection._validate_target(window_labels, len(positions))
            raw_window = raw_train.iloc[positions].drop(
                columns="isFraud", errors="ignore"
            )

            screen = RawFeatureScreen(
                raw_screen_strategy=raw_screen_strategy,
                top_v_features=top_v_features,
                preserve_features=preserve_features,
                seed=random_state,
            ).fit(raw_window, window_labels)

            screened_window = screen.transform(raw_window)

            features = FeatureEngineering(
                encoding=encoding,
                random_state=random_state,
                feature_groups=feature_groups,
                n_clusters=n_clusters,
            )
            encoded = features.fit_transform(screened_window, window_labels)

            selector = FeatureSelection(
                method=method,
                max_features=max_features,
                random_state=random_state,
                preserve_features=preserve_features,
                consensus_components=selection_consensus_components,
                device=selection_device,
            ).fit(encoded, window_labels)

        except ImportError as exc:
            report.update(status="skipped", detail=str(exc))
        except ValueError as exc:
            report.update(status="failed", detail=str(exc))
        else:
            identities = _stability_feature_identities(features, encoded.columns)
            canonical_selected = {
                identities[name] for name in selector.selected_features_
            }
            canonical_available = set(identities.values())

            report.update(
                status="fitted",
                feature_fit_rows=len(positions),
                raw_screen=screen.screening_report_,
                engineering={
                    "encoding": features.encoding_,
                    "feature_groups": list(features.feature_groups_),
                    "n_clusters": features.n_clusters,
                    "fitted_amount_mean": features.amount_stats_["mean"],
                    "output_features": features.output_features_.copy(),
                },
                selection=selector.selection_report_,
                selected_features=selector.selected_features_.copy(),
                canonical_selected_features=sorted(canonical_selected),
                feature_identity_mapping=identities,
            )

            selected_sets[float(fraction)] = canonical_selected
            available_sets[float(fraction)] = canonical_available

            for output_name, canonical_name in identities.items():
                inspection_names.setdefault(canonical_name, set()).add(output_name)

        reports.append(report)

    frequency = _build_frequency_table(
        selected_sets, available_sets, inspection_names
    )
    pairwise = _build_pairwise_table(selected_sets)

    return {
        "frequency": frequency,
        "pairwise": pairwise,
        "reports": reports,
        "identity_policy": {
            "ordinary_columns": "original feature name",
            "onehot_columns": (
                "source name plus SHA-256 of the frozen tagged category value; "
                "missing/unknown explicit"
            ),
            "cluster_columns": (
                "family level only; independent cluster labels and centroids "
                "are not aligned"
            ),
            "inspection_columns": (
                "original per-window names preserved in reports and frequency table"
            ),
        },
    }


def _build_frequency_table(selected_sets, available_sets, inspection_names):
    """Build the per-feature stability frequency table."""
    names = []
    if available_sets:
        names = sorted(set().union(*available_sets.values()))

    rows = []
    for name in names:
        selected_count = sum(name in s for s in selected_sets.values())
        available_count = sum(name in s for s in available_sets.values())
        rows.append(
            {
                "feature": name,
                "selection_count": selected_count,
                "selection_frequency": selected_count / len(selected_sets),
                "available_count": available_count,
                "frequency_when_available": selected_count / available_count,
                "inspection_features": sorted(inspection_names[name]),
            }
        )

    frequency = pd.DataFrame(
        rows,
        columns=[
            "feature",
            "selection_count",
            "selection_frequency",
            "available_count",
            "frequency_when_available",
            "inspection_features",
        ],
    )

    if not frequency.empty:
        frequency = frequency.sort_values(
            ["selection_frequency", "feature"], ascending=[False, True]
        ).reset_index(drop=True)

    return frequency


def _build_pairwise_table(selected_sets):
    """Build the pairwise Jaccard table between windows."""
    rows = []
    windows = list(selected_sets)

    for index, left in enumerate(windows):
        for right in windows[index + 1:]:
            intersection = selected_sets[left] & selected_sets[right]
            union = selected_sets[left] | selected_sets[right]
            rows.append(
                {
                    "left_fraction": left,
                    "right_fraction": right,
                    "intersection_count": len(intersection),
                    "union_count": len(union),
                    "jaccard": len(intersection) / len(union) if union else 1.0,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "left_fraction",
            "right_fraction",
            "intersection_count",
            "union_count",
            "jaccard",
        ],
    )
