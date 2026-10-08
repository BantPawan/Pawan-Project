"""Use a saved fraud pipeline to predict from raw joined transaction rows.

The loaded artifact owns feature construction, encoding, selection, imputation
and the decision threshold. Prediction reuses that fitted state for every
batch; it never learns new category mappings or preprocessing statistics.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .data import read_table
from .models import load_pipeline


class FraudPredictor:
    """Load one complete local or MLflow pipeline and reuse it for predictions."""

    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None

    def load_model(self, model_path=None):
        """Load a local pipeline file or a ``models:/`` / ``runs:/`` MLflow URI.

        The artifact must carry raw-row transforms and a stored threshold, so
        labels follow the same policy that was verified during evaluation.
        Loading a plain estimator that was logged without the surrounding
        ``FraudPipeline`` is rejected up front.
        """
        path = model_path or self.model_path
        if not path:
            raise ValueError(
                "Provide a complete pipeline path or models:/ registry URI"
            )

        if str(path).startswith(("models:/", "runs:/")):
            import mlflow.sklearn

            self.model = mlflow.sklearn.load_model(str(path))
        else:
            self.model = load_pipeline(path)

        required = ("transform", "predict_proba", "predict", "threshold")
        missing = [name for name in required if not hasattr(self.model, name)]
        if missing:
            raise ValueError(
                "Artifact must be a complete FraudPipeline exposing "
                f"{list(required)}; missing {missing}. A plain estimator "
                "logged without its surrounding pipeline cannot be used here."
            )

        self.model_path = path
        return self

    def preprocess_new_data(self, new_data):
        """Inspect numeric features using the model's already fitted transforms."""
        if self.model is None:
            raise ValueError("Call load_model first")

        return self.model.transform(new_data)

    def predict(self, new_data, return_probabilities=True):
        """Return fraud probabilities or labels using the artifact's saved threshold."""
        if self.model is None:
            raise ValueError("Call load_model first")

        if return_probabilities:
            return self.model.predict_proba(new_data)[:, 1]

        return self.model.predict(new_data)

    def predict_batch(self, data_path, output_path=None):
        """Score a raw DataFrame or CSV/Parquet batch and optionally save results.

        Every row keeps its TransactionID, fraud probability and predicted
        label. Probabilities are computed once and thresholded with the
        artifact's stored policy. Existing output files are protected from
        accidental replacement.
        """
        if self.model is None:
            raise ValueError("Call load_model first")

        data = (
            data_path.copy()
            if isinstance(data_path, pd.DataFrame)
            else read_table(data_path)
        )

        probabilities = self.model.predict_proba(data)[:, 1]
        predictions = (probabilities >= self.model.threshold).astype("int64")

        results = pd.DataFrame(
            {
                "TransactionID": data["TransactionID"],
                "isFraud_probability": probabilities,
                "isFraud_prediction": predictions,
            }
        )

        if output_path:
            path = Path(output_path)
            suffix = path.suffix.lower()
            if suffix not in (".csv", ".parquet"):
                raise ValueError(
                    "Prediction output must use a .csv or .parquet extension."
                )

            path.parent.mkdir(parents=True, exist_ok=True)

            if path.exists():
                raise FileExistsError(f"Prediction output already exists: {path}")

            if suffix == ".parquet":
                results.to_parquet(path, index=False)
            else:
                results.to_csv(path, index=False)

        return results

    def evaluate_prediction_quality(self, predictions, threshold=None):
        """Return the fraction of probabilities that would trigger a fraud alert.

        The historical method name is kept for compatibility. This measures
        alert volume only; observed fraud labels are needed to measure
        accuracy, precision or recall. Defaults to the loaded model's threshold.
        """
        if self.model is None:
            raise ValueError("Call load_model first")

        if threshold is None:
            threshold = self.model.threshold

        alert_flags = np.asarray(predictions) >= threshold
        return float(np.mean(alert_flags))