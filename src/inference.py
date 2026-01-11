"""Inference and evaluation utilities for the fraud RandomForest model bundle."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from config import FEATURES, TARGET_COL
from src.preprocessing import transform_x
from src.training import ModelBundle, TARGET_MAP


def predict_dataframe(bundle: ModelBundle, df: pd.DataFrame) -> pd.DataFrame:
    """Score a dataframe and return it with prediction columns appended."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for prediction: {missing}")

    x_raw = df[FEATURES].copy()
    x = transform_x(x_raw, bundle.preprocess)

    probas = bundle.model.predict_proba(x)
    preds = bundle.model.predict(x)

    out = df.copy()
    out["predicted_value"] = preds.astype(int)
    out["probability_of_value_0"] = probas[:, 0]
    out["probability_of_value_1"] = probas[:, 1]
    return out


def evaluate_on_labeled_csv(bundle: ModelBundle, df: pd.DataFrame) -> dict[str, Any]:
    """Evaluate a bundle on labeled data containing the target column."""
    missing = [c for c in FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for evaluation: {missing}")

    y = df[TARGET_COL].map(str).map(TARGET_MAP)
    mask = ~y.isnull()
    if int(mask.sum()) == 0:
        raise ValueError(
            "No valid target values found in test data "
            "(expected 0/1 in 'Is Fraudulent')."
        )

    y_true = y.loc[mask].astype(np.int64).to_numpy()

    x_raw = df.loc[mask, FEATURES].copy()
    x = transform_x(x_raw, bundle.preprocess)

    probas = bundle.model.predict_proba(x)[:, 1]
    y_pred = (probas >= 0.5).astype(int)

    auc = roc_auc_score(y_true, probas)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, digits=4)

    return {
        "auc": float(auc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n": int(len(y_true)),
        "positive_rate_true": float(y_true.mean()),
        "positive_rate_pred": float(y_pred.mean()),
        "confusion_matrix": cm,
        "classification_report": report,
    }
