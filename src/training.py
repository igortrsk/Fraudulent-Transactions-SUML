"""Training utilities for the fraud RandomForest model.

This module:
- Loads a labeled CSV dataset.
- Builds X/y arrays with target mapping and row filtering.
- Trains a RandomForest model with preprocessing fitted on training split only.
- Returns a `ModelBundle` for later inference/evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from config import FEATURES, TARGET_COL
from src.preprocessing import PreprocessArtifacts, fit_preprocess, transform_x as transform_X


TARGET_MAP = {"0": 0, "1": 1}


@dataclass
class ModelBundle:
    """A trained model packaged with preprocessing artifacts and feature metadata.

    Attributes:
        model: The trained `RandomForestClassifier`.
        preprocess: Preprocessing artifacts required for consistent inference.
        features_raw: Feature names expected as input (before preprocessing).
        features_after_preprocess: Column names produced after preprocessing.
    """
    model: RandomForestClassifier
    preprocess: PreprocessArtifacts
    features_raw: list[str]
    features_after_preprocess: list[str]


def load_csv_dataset(path: Path) -> pd.DataFrame:
    """Load a labeled CSV dataset and validate required columns.

    Args:
        path: Path to the CSV file.

    Returns:
        A dataframe containing only `config.FEATURES` and `config.TARGET_COL`.

    Raises:
        ValueError: If the CSV does not contain required columns.
    """
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[FEATURES + [TARGET_COL]].copy()


def build_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build feature dataframe X and target array y from a labeled dataframe.

    The target column is mapped using `TARGET_MAP` after converting to string.
    Rows with invalid or missing targets are filtered out.

    Args:
        df: Input dataframe containing features and target column.

    Returns:
        A tuple (X, y) where:
        - X is a dataframe of features for valid rows,
        - y is a NumPy array of int64 labels.
    """
    y = df[TARGET_COL].map(str).map(TARGET_MAP)
    df = df.drop(columns=[TARGET_COL])
    mask = ~y.isnull()
    X = df.loc[mask].copy()
    y = y.loc[mask].astype(np.int64).to_numpy()
    return X, y


def train_random_forest(
    train_path: Path,
    test_size: float = 0.2,
    seed: int = 1337,
) -> Tuple[ModelBundle, Dict[str, Any]]:
    """Train a RandomForest model and return a bundled artifact plus metrics.

    Workflow:
    - Load dataset and build X/y.
    - Split into train/test.
    - Fit preprocessing on training split only.
    - Train a RandomForest classifier.
    - Compute AUC on the test split.

    Args:
        train_path: Path to the training CSV containing features and target column.
        test_size: Proportion of data to use as the test split.
        seed: Random seed for reproducibility of the train/test split.

    Returns:
        A tuple (bundle, metrics) where:
        - bundle is a `ModelBundle` containing the trained model and preprocessing artifacts,
        - metrics is a dict with test AUC and basic dataset statistics.
    """
    df = load_csv_dataset(train_path)
    X, y = build_xy(df)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, shuffle=True
    )

    art = fit_preprocess(X_train_raw)
    X_train = transform_X(X_train_raw, art)
    X_test = transform_X(X_test_raw, art)

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=1337,
        max_depth=6,
        min_samples_leaf=1,
        verbose=2,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    probas = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probas)

    metrics = {
        "auc": float(auc),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "positive_rate_true_test": float(y_test.mean()),
        "positive_rate_pred_test": float((probas >= 0.5).mean()),
    }

    bundle = ModelBundle(
        model=clf,
        preprocess=art,
        features_raw=FEATURES,
        features_after_preprocess=list(X_train.columns),
    )
    return bundle, metrics
