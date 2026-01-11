from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class PreprocessArtifacts:
    ta_min: float
    ta_max: float
    aad_mean: float
    aad_std: float
    ta_impute: float
    aad_impute: float


def fit_preprocess(train_x: pd.DataFrame) -> PreprocessArtifacts:
    """Fit preprocessing parameters on training features only."""
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
    """Apply imputation and scaling to features."""
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
