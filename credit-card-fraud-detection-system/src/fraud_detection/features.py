"""Raw screening and feature engineering inside the sklearn Pipeline.

Two ``sklearn`` transformers, both fitted inside each chronological fold so
that derived statistics and encoders never leak across partitions:

* ``RawFeatureScreen`` — decides which raw columns proceed to engineering.
  ``"all"`` keeps everything; ``"strategic"`` ranks the ``V`` columns by
  their training-only correlation with the label and keeps the top ``N``,
  always preserving the caller's declared columns.

* ``FeatureEngineering`` — builds amount, temporal, product, card, count,
  email, identity, missingness, composite and cluster features. Fits the
  training statistics that later rows reuse (amount quantiles, card1
  frequency, C z-score parameters, categorical code maps, KMeans centres).

Both stages expose the attributes the rest of the workflow relies on:

* ``RawFeatureScreen.selected_features_`` — raw columns kept.
* ``RawFeatureScreen.screening_report_`` — how the screen decided.

* ``FeatureEngineering.input_columns_`` — raw columns the transformer saw.
* ``FeatureEngineering.output_features_`` — numeric output schema.
* ``FeatureEngineering.amount_stats_`` — training Amount statistics
  (``q25``, ``q75``, ``median``, ``mean``, ``std``, ``iqr``) reused by the
  evaluation diagnostics for amount slices.
* ``FeatureEngineering.feature_groups_`` — groups that produced features.
* ``FeatureEngineering.encoding_`` — resolved encoding policy.
* ``FeatureEngineering.onehot_columns_`` — ``{source: {code: output_name}}``
  for one-hot encoded categories, used by the stability analysis.
* ``FeatureEngineering.category_maps_`` — ``{source: {category: code}}``
  frozen category dictionary for each label- or one-hot-encoded column.

The module never uses the target to build a feature. ``RawFeatureScreen``
uses the label only for its optional ``strategic`` V ranking, which is a
training-fold decision like any other model coefficient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .utils import column_view


# ---------------------------------------------------------------------
# Small value mappers shared by the email and identity builders
# ---------------------------------------------------------------------

_EMAIL_FAMILIES = (
    ("gmail", "gmail"),
    ("yahoo", "yahoo"),
    ("hotmail", "hotmail"),
    ("protonmail", "protonmail"),
    ("outlook", "outlook"),
    ("aol", "aol"),
    ("icloud", "icloud"),
)

_OS_FAMILIES = (
    ("windows", "Windows"),
    ("android", "Android"),
    ("ios", "iOS"),
    ("iphone", "iOS"),
    ("mac", "Mac"),
    ("linux", "Linux"),
)

_BROWSER_FAMILIES = (
    ("samsung", "Samsung"),
    ("edge", "IE/Edge"),
    ("chrome", "Chrome"),
    ("safari", "Safari"),
    ("firefox", "Firefox"),
    ("android", "Android Browser"),
)

_POPULAR_EMAILS = (
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "aol.com",
    "icloud.com",
)

_IDENTIFIER_COLUMNS = ("TransactionID", "isFraud")


def _email_provider(value):
    if pd.isna(value):
        return np.nan
    text = str(value).lower()
    for token, family in _EMAIL_FAMILIES:
        if token in text:
            return family
    return "other"


def _os_family(value):
    if pd.isna(value):
        return np.nan
    text = str(value).lower()
    for token, family in _OS_FAMILIES:
        if token in text:
            return family
    return "Other"


def _browser_family(value):
    if pd.isna(value):
        return np.nan
    text = str(value).lower()
    for token, family in _BROWSER_FAMILIES:
        if token in text:
            return family
    return "Other"


# ---------------------------------------------------------------------
# Raw screening
# ---------------------------------------------------------------------

class RawFeatureScreen(BaseEstimator, TransformerMixin):
    """Choose which raw columns proceed to engineering.

    ``"all"`` keeps every column. ``"strategic"`` ranks ``V`` columns by the
    absolute correlation with the training label, keeps the top
    ``top_v_features`` after dropping heavily missing ones, and always
    retains the non-``V`` columns plus anything in ``preserve_features``.
    The screen runs on training rows only and never touches the target at
    transform time.
    """

    STRATEGIES = ("all", "strategic")
    MAX_MISSING_FRACTION = 0.9
    DEFAULT_TOP_V = 150

    def __init__(
        self,
        raw_screen_strategy="all",
        top_v_features=None,
        preserve_features=(),
        seed=42,
    ):
        self.raw_screen_strategy = raw_screen_strategy
        self.top_v_features = top_v_features
        self.preserve_features = preserve_features
        self.seed = seed

    def __setstate__(self, state):
        """Keep earlier saved screens loadable."""
        super().__setstate__(state)
        for name, value in (
            ("raw_screen_strategy", "all"),
            ("top_v_features", None),
            ("preserve_features", ()),
            ("seed", 42),
        ):
            if not hasattr(self, name):
                setattr(self, name, value)

    # -- validation --

    @staticmethod
    def _validate_frame(frame):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("RawFeatureScreen expects a pandas DataFrame.")
        if frame.columns.has_duplicates:
            raise ValueError("RawFeatureScreen input has duplicate column names.")
        if "TransactionID" not in frame.columns:
            raise ValueError("RawFeatureScreen requires TransactionID.")
        return frame

    # -- fit --

    def fit(self, X, y=None):
        if self.raw_screen_strategy not in self.STRATEGIES:
            raise ValueError(
                f"raw_screen_strategy must be one of {self.STRATEGIES}."
            )
        if self.top_v_features is not None and (
            isinstance(self.top_v_features, bool)
            or not isinstance(self.top_v_features, (int, np.integer))
            or self.top_v_features < 1
        ):
            raise ValueError("top_v_features must be a positive integer or None.")

        X = self._validate_frame(X)
        candidates = [c for c in X.columns if c not in _IDENTIFIER_COLUMNS]

        self.input_columns_ = list(X.columns)

        if self.raw_screen_strategy == "all":
            self.selected_features_ = candidates
            self.screening_report_ = {
                "strategy": "all",
                "input_columns": len(self.input_columns_),
                "selected_columns": len(self.selected_features_),
                "dropped_columns": [],
            }
            return self

        # Strategic V ranking needs the training labels.
        if y is None:
            raise ValueError("Strategic V screening requires training labels.")

        labels = pd.Series(np.asarray(y), index=X.index).astype(float)

        v_columns = [c for c in candidates if c.startswith("V")]
        non_v_columns = [c for c in candidates if c not in v_columns]
        preserved = {c for c in self.preserve_features if c in candidates}

        top_n = self.top_v_features or self.DEFAULT_TOP_V
        usable_v: list[str] = []
        selected_v: list[str] = []

        if v_columns:
            numeric_v = X[v_columns].apply(pd.to_numeric, errors="coerce")
            missing_fraction = numeric_v.isna().mean()
            usable_v = missing_fraction[
                missing_fraction < self.MAX_MISSING_FRACTION
            ].index.tolist()

            if usable_v:
                correlations = (
                    numeric_v[usable_v]
                    .corrwith(labels)
                    .abs()
                    .sort_values(ascending=False)
                )
                selected_v = correlations.head(top_n).index.tolist()

        selected = non_v_columns + selected_v
        for column in preserved:
            if column not in selected:
                selected.append(column)

        self.selected_features_ = selected
        self.screening_report_ = {
            "strategy": "strategic",
            "input_columns": len(self.input_columns_),
            "selected_columns": len(selected),
            "dropped_columns": [
                c for c in self.input_columns_ if c not in selected
            ],
            "v_candidates": len(v_columns),
            "v_usable": len(usable_v),
            "v_selected": len(selected_v),
            "top_v_requested": self.top_v_features,
            "max_missing_fraction": self.MAX_MISSING_FRACTION,
        }
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_features_")
        X = self._validate_frame(X)
        missing = [c for c in self.selected_features_ if c not in X.columns]
        if missing:
            raise ValueError(
                f"RawFeatureScreen transform is missing required columns: {missing}"
            )
        # Downstream engineering replaces columns instead of editing shared values.
        return column_view(X, self.selected_features_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_features_")
        return np.asarray(self.selected_features_, dtype=object)


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Build the engineered numeric frame used by every downstream stage.

    ``encoding="legacy"`` label-encodes low-cardinality categoricals to
    ``{source}_encoded``; ``encoding="onehot"`` instead emits
    ``{source}_onehot_{code}`` indicator columns. Both modes frequency-encode
    high-cardinality categoricals to ``{source}_freq``. Code ``0`` is
    reserved for missing; unknown categories at transform time receive
    ``-1``.

    ``feature_groups="core"`` drops ``V`` columns before engineering.
    ``n_clusters > 0`` additionally fits KMeans on amount, time and card
    frequency features and emits an integer ``behavioral_cluster`` column.
    """

    CATEGORICAL_THRESHOLD = 10
    ENCODINGS = ("legacy", "onehot")
    FEATURE_GROUPS = ("all", "core")
    CLUSTER_SOURCES = (
        "TransactionAmt_log",
        "amount_zscore",
        "elapsed_hour_phase",
        "elapsed_day_phase",
        "card1_frequency",
    )

    def __init__(
        self,
        encoding="legacy",
        random_state=42,
        feature_groups="all",
        n_clusters=0,
    ):
        self.encoding = encoding
        self.random_state = random_state
        self.feature_groups = feature_groups
        self.n_clusters = n_clusters

    def __setstate__(self, state):
        """Keep earlier saved engineering loadable."""
        super().__setstate__(state)
        for name, value in (
            ("encoding", "legacy"),
            ("feature_groups", "all"),
            ("n_clusters", 0),
            ("random_state", 42),
        ):
            if not hasattr(self, name):
                setattr(self, name, value)

    # -----------------------------------------------------------------
    # Validation helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_frame(frame):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("FeatureEngineering expects a pandas DataFrame.")
        if frame.columns.has_duplicates:
            raise ValueError("FeatureEngineering input has duplicate columns.")
        return frame

    def _validate_options(self):
        if self.encoding not in self.ENCODINGS:
            raise ValueError(
                f"encoding must be one of {self.ENCODINGS}, got {self.encoding!r}."
            )
        if self.feature_groups not in self.FEATURE_GROUPS:
            raise ValueError(
                f"feature_groups must be one of {self.FEATURE_GROUPS}."
            )
        if (
            isinstance(self.n_clusters, bool)
            or not isinstance(self.n_clusters, (int, np.integer))
            or self.n_clusters < 0
        ):
            raise ValueError("n_clusters must be a non-negative integer.")

    def _columns_in_scope(self, X):
        if self.feature_groups == "core":
            return [c for c in X.columns if not c.startswith("V")]
        return list(X.columns)

    # -----------------------------------------------------------------
    # Feature builders
    # -----------------------------------------------------------------

    @staticmethod
    def _add_amount_features(frame, stats):
        amt = frame["TransactionAmt"]
        lower = stats["q25"] - 1.5 * stats["iqr"]
        upper = stats["q75"] + 1.5 * stats["iqr"]

        frame["TransactionAmt_log"] = np.log1p(amt.clip(lower=0))
        frame["is_amount_outlier"] = (
            (amt < lower) | (amt > upper)
        ).astype(int)
        frame["amount_to_median"] = amt / stats["median"]
        frame["amount_to_q75"] = amt / stats["q75"]
        frame["amount_zscore"] = (amt - stats["mean"]) / stats["std"]
        return frame

    @staticmethod
    def _add_temporal_features(frame):
        hour_phase = (frame["TransactionDT"] // 3600) % 24
        day_phase = (frame["TransactionDT"] // 86400) % 7

        frame["elapsed_hour_phase"] = hour_phase
        frame["elapsed_day_phase"] = day_phase
        frame["elapsed_hour_sin"] = np.sin(2 * np.pi * hour_phase / 24)
        frame["elapsed_hour_cos"] = np.cos(2 * np.pi * hour_phase / 24)
        frame["elapsed_week_sin"] = np.sin(2 * np.pi * day_phase / 7)
        frame["elapsed_week_cos"] = np.cos(2 * np.pi * day_phase / 7)
        return frame

    @staticmethod
    def _add_product_features(frame):
        if "ProductCD" not in frame.columns:
            return frame
        values = frame["ProductCD"].astype("string").str.upper()
        for code in ("W", "C", "H", "R", "S"):
            frame[f"is_product_{code}"] = (
                values.eq(code).fillna(False).astype(int)
            )
        frame["product_missing"] = values.isna().astype(int)
        return frame

    @staticmethod
    def _add_card_features(frame, card1_freq_map):
        if "card1" in frame.columns:
            freq = frame["card1"].map(card1_freq_map).fillna(0)
            frame["card1_frequency"] = freq
            frame["is_single_use_card"] = (freq == 1).astype(int)
            frame["is_high_freq_card"] = (freq > 10).astype(int)

        if "card6" in frame.columns:
            values = frame["card6"].astype("string").str.lower()
            frame["is_credit_card"] = values.eq("credit").fillna(False).astype(int)
            frame["is_debit_card"] = values.eq("debit").fillna(False).astype(int)
            frame["is_debit_or_credit"] = (
                values.isin(["credit", "debit"]).fillna(False).astype(int)
            )
        return frame

    @staticmethod
    def _add_count_features(frame, c_stats):
        outlier_flags = []
        for column, stats in c_stats.items():
            if column not in frame.columns:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            lower = stats["q25"] - 1.5 * stats["iqr"]
            upper = stats["q75"] + 1.5 * stats["iqr"]

            flag = ((values < lower) | (values > upper)).astype(int)
            frame[f"{column}_outlier"] = flag
            frame[f"{column}_zscore"] = (values - stats["mean"]) / stats["std"]
            if stats["skew"] > 1:
                frame[f"{column}_log"] = np.log1p(values.clip(lower=0))
            outlier_flags.append(flag)

        if outlier_flags:
            frame["any_C_outlier"] = (
                pd.concat(outlier_flags, axis=1).max(axis=1)
            )
        return frame

    @staticmethod
    def _add_email_features(frame):
        for column, suffix in (("P_emaildomain", "P"), ("R_emaildomain", "R")):
            if column not in frame.columns:
                continue
            values = frame[column].astype("string").str.lower()
            frame[f"email_provider{suffix}"] = values.map(_email_provider)
            frame[f"is_missing{suffix}"] = values.isna().astype(int)
            frame[f"is_protonmail{suffix}"] = (
                values.eq("protonmail.com").fillna(False).astype(int)
            )
            frame[f"is_mail_com{suffix}"] = (
                values.eq("mail.com").fillna(False).astype(int)
            )
            frame[f"is_popular{suffix}"] = (
                values.isin(_POPULAR_EMAILS).fillna(False).astype(int)
            )
            frame[f"is_anonymous{suffix}"] = (
                values.eq("anonymous.com").fillna(False).astype(int)
            )

        if "P_emaildomain" in frame.columns and "R_emaildomain" in frame.columns:
            left = frame["P_emaildomain"].astype("string").str.lower()
            right = frame["R_emaildomain"].astype("string").str.lower()
            frame["email_match"] = left.eq(right).fillna(False).astype(int)
        return frame

    @staticmethod
    def _add_identity_features(frame):
        if "id_30" in frame.columns:
            frame["OS"] = frame["id_30"].map(_os_family)
            frame["OS_missing"] = frame["id_30"].isna().astype(int)
        if "id_31" in frame.columns:
            frame["Browser"] = frame["id_31"].map(_browser_family)
            frame["browser_missing"] = frame["id_31"].isna().astype(int)
        if "DeviceType" in frame.columns:
            values = frame["DeviceType"].astype("string").str.lower()
            frame["is_mobile_device"] = (
                values.eq("mobile").fillna(False).astype(int)
            )
            frame["is_desktop_device"] = (
                values.eq("desktop").fillna(False).astype(int)
            )
            frame["DeviceType_missing"] = values.isna().astype(int)
        if "DeviceInfo" in frame.columns:
            frame["device_os"] = frame["DeviceInfo"].map(_os_family)
            frame["device_os_missing"] = frame["DeviceInfo"].isna().astype(int)

        for column in ("id_35", "id_36", "id_37", "id_38"):
            if column not in frame.columns:
                continue
            values = frame[column].astype("string")
            frame[f"{column}_is_F"] = values.eq("F").fillna(False).astype(int)
            frame[f"{column}_is_T"] = values.eq("T").fillna(False).astype(int)
            frame[f"{column}_missing"] = frame[column].isna().astype(int)

        if "id_33" in frame.columns:
            parts = (
                frame["id_33"]
                .astype("string")
                .str.extract(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$")
            )
            frame["screen_width"] = pd.to_numeric(parts[0], errors="coerce")
            frame["screen_height"] = pd.to_numeric(parts[1], errors="coerce")
            frame["screen_area"] = frame["screen_width"] * frame["screen_height"]
            frame["resolution_missing"] = frame["id_33"].isna().astype(int)
        return frame

    @staticmethod
    def _add_missingness_features(frame):
        groups = {
            "D_features": [
                c for c in frame.columns if c.startswith("D") and len(c) <= 3
            ],
            "card_features": [
                c for c in frame.columns if c.startswith("card")
            ],
            "identity_features": [
                c for c in frame.columns if c.startswith("id_")
            ],
            "email_features": [
                c for c in ("P_emaildomain", "R_emaildomain") if c in frame.columns
            ],
            "M_features": [
                c for c in frame.columns if c.startswith("M") and len(c) == 2
            ],
            "address_features": [
                c for c in ("addr1", "addr2") if c in frame.columns
            ],
        }
        for name, columns in groups.items():
            if columns:
                frame[f"missing_{name}"] = frame[columns].isna().mean(axis=1)

        predictors = [
            c for c in frame.columns if c not in _IDENTIFIER_COLUMNS
        ]
        if predictors:
            frame["missing_overall"] = frame[predictors].isna().mean(axis=1)
            frame["is_high_missing"] = (
                frame["missing_overall"] > 0.5
            ).astype(int)
        return frame

    @staticmethod
    def _add_composite_score(frame):
        inputs = [
            c for c in (
                "is_amount_outlier",
                "is_single_use_card",
                "missing_identity_features",
                "is_high_missing",
            ) if c in frame.columns
        ]
        if inputs:
            frame["composite_anomaly_score"] = (
                frame[inputs].fillna(0).clip(0, 1).mean(axis=1)
            )
        return frame

    def _add_cluster_features(self, frame):
        if self.cluster_model_ is None or not self.cluster_columns_:
            frame["behavioral_cluster"] = 0
            return frame

        values = frame[self.cluster_columns_].apply(
            pd.to_numeric, errors="coerce"
        )
        if len(values):
            values = values.fillna(self.cluster_medians_)
        else:
            values = values.fillna(0.0)

        labels = self.cluster_model_.predict(
            self.cluster_scaler_.transform(values)
        )
        frame["behavioral_cluster"] = labels.astype(int)
        return frame

    # -----------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------

    def fit(self, X, y=None):
        """Learn statistics, encoders and clusters from the training rows.

        ``y`` is accepted for sklearn Pipeline compatibility and is not used
        to build any feature.
        """
        self._validate_options()
        X = self._validate_frame(X)

        for required in ("TransactionDT", "TransactionAmt"):
            if required not in X.columns:
                raise ValueError(f"FeatureEngineering requires {required}.")

        self.input_columns_ = list(X.columns)
        self.encoding_ = self.encoding
        self.feature_groups_ = (self.feature_groups,)

        self._fit_amount_stats(X)
        self._fit_card_frequency(X)
        self._fit_count_stats(X)

        base = self._build_base_features(X)

        if self.n_clusters > 0:
            self._fit_clusters(base)
        else:
            self.cluster_scaler_ = None
            self.cluster_model_ = None
            self.cluster_columns_ = []
            self.cluster_medians_ = pd.Series(dtype="float64")

        base = self._add_cluster_features(base)
        self._fit_encoders(base)

        # Discover the output schema by running the full transform on the
        # training rows once. Later transforms reindex to this schema.
        transformed = self._transform_validated(X)
        self.output_features_ = transformed.columns.tolist()
        return self

    def _fit_amount_stats(self, X):
        amount = pd.to_numeric(X["TransactionAmt"], errors="coerce")
        if amount.notna().any():
            stats = {
                "median": float(amount.median()),
                "q25": float(amount.quantile(0.25)),
                "q75": float(amount.quantile(0.75)),
                "mean": float(amount.mean()),
                "std": float(amount.std(ddof=0)),
            }
        else:
            stats = {"median": 0.0, "q25": 0.0, "q75": 0.0, "mean": 0.0, "std": 1.0}

        stats["iqr"] = stats["q75"] - stats["q25"]

        if not np.isfinite(stats["std"]) or stats["std"] == 0:
            stats["std"] = 1.0
        for name in ("median", "q75"):
            if not np.isfinite(stats[name]) or stats[name] == 0:
                stats[name] = 1.0

        self.amount_stats_ = stats

    def _fit_card_frequency(self, X):
        if "card1" in X.columns:
            self.card1_freq_map_ = (
                X["card1"].value_counts(dropna=True).to_dict()
            )
        else:
            self.card1_freq_map_ = {}

    def _fit_count_stats(self, X):
        self.c_stats_ = {}
        for index in range(1, 15):
            column = f"C{index}"
            if column not in X.columns:
                continue
            values = pd.to_numeric(X[column], errors="coerce").dropna()
            if values.empty:
                continue
            q25, q75 = values.quantile([0.25, 0.75])
            std = values.std(ddof=0)
            if not np.isfinite(std) or std == 0:
                std = 1.0
            self.c_stats_[column] = {
                "q25": q25,
                "q75": q75,
                "iqr": q75 - q25,
                "mean": values.mean(),
                "std": std,
                "skew": values.skew() if len(values) >= 3 else 0.0,
            }

    def _fit_clusters(self, base):
        columns = [c for c in self.CLUSTER_SOURCES if c in base.columns]
        if len(columns) < 3 or len(base) < self.n_clusters:
            self.cluster_scaler_ = None
            self.cluster_model_ = None
            self.cluster_columns_ = []
            self.cluster_medians_ = pd.Series(dtype="float64")
            return

        values = base[columns].apply(pd.to_numeric, errors="coerce")
        self.cluster_medians_ = values.median().fillna(0.0)
        filled = values.fillna(self.cluster_medians_)

        self.cluster_scaler_ = StandardScaler().fit(filled)
        self.cluster_model_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        ).fit(self.cluster_scaler_.transform(filled))
        self.cluster_columns_ = columns

    def _fit_encoders(self, frame):
        self.label_maps_ = {}
        self.freq_maps_ = {}
        self.category_maps_ = {}
        self.onehot_columns_ = {}

        for column in frame.columns:
            if pd.api.types.is_numeric_dtype(frame[column]):
                continue

            values = frame[column].astype("string").fillna("__MISSING__")
            unique_values = list(dict.fromkeys(values.tolist()))

            if len(unique_values) <= self.CATEGORICAL_THRESHOLD:
                ordered = ["__MISSING__"] + sorted(
                    value for value in unique_values if value != "__MISSING__"
                )
                code_map = {value: index for index, value in enumerate(ordered)}
                self.category_maps_[column] = dict(code_map)

                if self.encoding_ == "legacy":
                    self.label_maps_[column] = dict(code_map)
                else:
                    self.label_maps_[column] = dict(code_map)
                    self.onehot_columns_[column] = {
                        code: f"{column}_onehot_{code}"
                        for code in code_map.values()
                    }
            else:
                self.freq_maps_[column] = values.value_counts().to_dict()

    # -----------------------------------------------------------------
    # Transform
    # -----------------------------------------------------------------

    def transform(self, X):
        check_is_fitted(self, "amount_stats_")
        X = self._validate_frame(X)
        return self._transform_validated(X)

    def _transform_validated(self, X):
        frame = self._build_base_features(X)
        frame = self._add_cluster_features(frame)
        frame = self._apply_encoders(frame)

        non_numeric = [
            c for c in frame.columns
            if not pd.api.types.is_numeric_dtype(frame[c])
        ]
        if non_numeric:
            frame = frame.drop(columns=non_numeric)
        frame = frame.astype("float64", copy=False)

        if hasattr(self, "output_features_"):
            missing = [
                c for c in self.output_features_ if c not in frame.columns
            ]
            if missing:
                for column in missing:
                    frame[column] = np.nan
            frame = frame[self.output_features_]

        return frame

    def _build_base_features(self, X):
        columns = self._columns_in_scope(X)
        columns = [c for c in columns if c not in _IDENTIFIER_COLUMNS]
        # Keep an independent schema while sharing unchanged raw column buffers.
        frame = column_view(X, columns)

        # Required columns must survive the core/all filter.
        for required in ("TransactionDT", "TransactionAmt"):
            if required not in frame.columns:
                frame[required] = X[required]

        frame["TransactionAmt"] = pd.to_numeric(
            frame["TransactionAmt"], errors="coerce"
        )
        frame["TransactionDT"] = pd.to_numeric(
            frame["TransactionDT"], errors="coerce"
        )

        frame = self._add_amount_features(frame, self.amount_stats_)
        frame = self._add_temporal_features(frame)
        frame = self._add_product_features(frame)
        frame = self._add_card_features(frame, self.card1_freq_map_)
        frame = self._add_count_features(frame, self.c_stats_)
        frame = self._add_email_features(frame)
        frame = self._add_identity_features(frame)
        frame = self._add_missingness_features(frame)
        frame = self._add_composite_score(frame)
        return frame

    def _apply_encoders(self, frame):
        # Encoders only add or replace whole columns; existing values stay read-only.
        frame = frame.copy(deep=False)

        for source, code_map in self.label_maps_.items():
            if source not in frame.columns:
                continue
            values = frame[source].astype("string").fillna("__MISSING__")
            codes = values.map(code_map).fillna(-1).astype(int)

            if self.encoding_ == "legacy":
                frame[f"{source}_encoded"] = codes
            else:
                outputs = self.onehot_columns_.get(source, {})
                for code, output_name in outputs.items():
                    frame[output_name] = (codes == code).astype(int)

        for source, freq_map in self.freq_maps_.items():
            if source not in frame.columns:
                continue
            values = frame[source].astype("string").fillna("__MISSING__")
            frame[f"{source}_freq"] = (
                values.map(freq_map).fillna(0.0).astype(float)
            )

        return frame

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "output_features_")
        return np.asarray(self.output_features_, dtype=object)
