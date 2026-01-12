"""Preprocessing utilities for the fraud model.

This module defines preprocessing artifacts computed from training data and functions
to apply the same transformations during inference.

Current preprocessing:
- Imputation of missing values using training means.
- Min-max scaling for "Transaction Amount" (or drop if constant).
- Standardization (z-score) for "Account Age Days".
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class PreprocessArtifacts:
    """Artifacts needed to reproduce training-time preprocessing during inference.

    Attributes:
        ta_min: Minimum of imputed "Transaction Amount" from training data.
        ta_max: Maximum of imputed "Transaction Amount" from training data.
        aad_mean: Mean of imputed "Account Age Days" from training data.
        aad_std: Standard deviation of imputed "Account Age Days" from training data.
        ta_impute: Mean used to impute missing "Transaction Amount".
        aad_impute: Mean used to impute missing "Account Age Days".
    """
    ta_min: float
    ta_max: float
    aad_mean: float
    aad_std: float
    ta_impute: float
    aad_impute: float


def fit_preprocess(train_x: pd.DataFrame) -> PreprocessArtifacts:
    """Fit preprocessing parameters on training features.

    Computes imputation values and scaling parameters based on training data only.
    Returned artifacts should be stored with the trained model to ensure consistent
    transformations during inference.

    Args:
        train_x: Training feature dataframe.

    Returns:
        A `PreprocessArtifacts` instance containing learned preprocessing parameters.
    """
    ta_impute = float(train_x["Transaction Amount"].mean())
    aad_impute = float(train_x["Account Age Days"].mean())

    filled = train_x.copy()
    filled["Transaction Amount"] = filled["Transaction Amount"].fillna(ta_impute).astype("float64")
    filled["Account Age Days"] = filled["Account Age Days"].fillna(aad_impute).astype("float64")

    ta_min = float(filled["Transaction Amount"].min())
    ta_max = float(filled["Transaction Amount"].max())

    aad_mean = float(filled["Account Age Days"].mean())
    aad_std = float(filled["Account Age Days"].std(ddof=1)) or 1.0

    return PreprocessArtifacts(
        ta_min=ta_min,
        ta_max=ta_max,
        aad_mean=aad_mean,
        aad_std=aad_std,
        ta_impute=ta_impute,
        aad_impute=aad_impute,
    )


def transform_x(df_x: pd.DataFrame, art: PreprocessArtifacts) -> pd.DataFrame:
    """Apply imputation and feature scaling to input features.

    Steps:
    1) Impute missing values for "Transaction Amount" and "Account Age Days".
    2) Min-max scale "Transaction Amount" using training min/max.
       If the feature is constant (max == min), it is dropped.
    3) Standardize "Account Age Days" using training mean/std.

    Args:
        df_x: Raw feature dataframe to transform.
        art: Preprocessing artifacts learned from training data.

    Returns:
        A transformed dataframe ready for model consumption.
    """
    out = df_x.copy()

    out["Transaction Amount"] = out["Transaction Amount"].fillna(art.ta_impute).astype("float64")
    out["Account Age Days"] = out["Account Age Days"].fillna(art.aad_impute).astype("float64")

    ta_scale = art.ta_max - art.ta_min
    if ta_scale == 0:
        out = out.drop(columns=["Transaction Amount"])
    else:
        out["Transaction Amount"] = (out["Transaction Amount"] - art.ta_min) / ta_scale

    out["Account Age Days"] = (out["Account Age Days"] - art.aad_mean) / art.aad_std
    return out
