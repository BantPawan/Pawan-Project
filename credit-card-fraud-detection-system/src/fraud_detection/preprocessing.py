"""Fill missing numeric features and optionally scale them using training state.

FeatureEngineering handles raw categories, then feature selection chooses the
columns. This final step learns medians and scaling parameters inside the
fitted model pipeline, keeping validation and future data out of that fit.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from .utils import numeric_feature_view


class Preprocessor(TransformerMixin, BaseEstimator):
    """Freeze training medians and optional scaling for numeric feature frames.

    Raw categoricals are handled earlier by FeatureEngineering, so this step
    keeps imputation and scaling inside training folds. Columns that are
    entirely missing during training are filled with 0. All other missing
    cells use the training median. Output keeps the original column names,
    order and float64 dtype. ``scale=False`` suits tree models.
    ``scale_categorical=False`` leaves generated label codes, one-hot
    indicators and cluster IDs alone while scaling continuous and frequency
    features. The defaults preserve the original all-column scaling interface.
    """

    def __init__(self, scale=True, scale_categorical=True):
        self.scale = scale
        self.scale_categorical = scale_categorical

    @staticmethod
    def _categorical_output(name):
        """Recognize FeatureEngineering's categorical output namespace."""
        return (
            name.endswith("_encoded")
            or "_onehot_" in name
            or name == "behavioral_cluster"
        )

    @staticmethod
    def _numeric_frame(X):
        """Validate a numeric feature frame and treat infinities as missing."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Preprocessor expects a numeric pandas DataFrame.")
        if X.columns.has_duplicates:
            raise ValueError("Feature input contains duplicate column names.")
        if any(not pd.api.types.is_numeric_dtype(X[name]) for name in X):
            raise ValueError(
                "Preprocessor expects numeric features; encode raw categories first."
            )
        return numeric_feature_view(X)

    def fit(self, X, y=None):
        """Learn column order, medians and optional scaling from training features.

        ``y`` is accepted for sklearn Pipeline compatibility. Columns that are
        entirely missing during training are filled with 0. All other columns
        use their training median. Later transforms reuse these values instead
        of recomputing them.
        """
        values = self._numeric_frame(X)
        if values.empty or not len(values.columns):
            raise ValueError("Cannot fit preprocessing on an empty feature frame.")

        self.input_columns_ = values.columns.tolist()
        self.output_features_ = self.input_columns_.copy()
        self.feature_names_in_ = np.asarray(self.input_columns_, dtype=object)
        self.n_features_in_ = len(self.input_columns_)

        self.medians_ = values.median().fillna(0.0)

        self.categorical_columns_ = [
            name for name in self.input_columns_ if self._categorical_output(name)
        ]

        self.scaled_columns_ = [
            name
            for name in self.input_columns_
            if self.scale
            and (self.scale_categorical or name not in self.categorical_columns_)
        ]

        if self.scaled_columns_:
            scaling_values = values[self.scaled_columns_].fillna(self.medians_)
            self.scaler_ = StandardScaler().fit(scaling_values)
        else:
            self.scaler_ = None

        return self

    def transform(self, X):
        """Apply the frozen schema, median fill and optional scaling.

        Extra columns are ignored. Missing required engineered columns raise
        an explicit error, so an upstream schema change cannot silently alter
        inference.
        """
        check_is_fitted(self, "medians_")
        values = self._numeric_frame(X)

        missing = [name for name in self.input_columns_ if name not in values]
        if missing:
            raise ValueError(f"Missing engineered feature columns: {missing}")

        values = values.loc[:, self.input_columns_].fillna(self.medians_)

        if not values.empty and self.scaler_ is not None:
            # Fallback supports artifacts fitted before selective scaling.
            scaled_columns = getattr(self, "scaled_columns_", self.input_columns_)
            values.loc[:, scaled_columns] = self.scaler_.transform(
                values[scaled_columns]
            )

        return values.astype("float64", copy=False)

    def get_feature_names_out(self, input_features=None):
        """Return fitted output column names in their original training order."""
        check_is_fitted(self, "output_features_")
        return np.asarray(self.output_features_, dtype=object)
