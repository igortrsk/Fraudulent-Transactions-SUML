"""Streamlit UI for fraud prediction.

Provides two modes:
1) Manual input: single transaction prediction from Amount and Account Age Days.
2) CSV batch input: predictions for uploaded CSV with columns (id, amount, account_age).

The app loads a pickled `ModelBundle` and uses the same preprocessing + model
logic as the CLI inference utilities.
"""
from pathlib import Path
import io
import pandas as pd
import streamlit as st
from config import FEATURES
from src.inference import predict_dataframe

from app import load_bundle_pickle

MODEL_DEFAULT_PATH = "models/fraud_rf_bundle.pkl"

st.set_page_config(page_title="Ecomm Fraud Prediction", layout="wide")
st.title("Fraud prediction")

def load_model(model_path: str):
    """Load a pickled model bundle from the given path.

    Args:
        model_path: Path to the pickled `ModelBundle`.

    Returns:
        Loaded `ModelBundle`.
    """
    return load_bundle_pickle(Path(model_path))

def read_csv(file_bytes: bytes) -> pd.DataFrame:
    """Read a CSV file from raw bytes into a dataframe.

    Args:
        file_bytes: CSV content as bytes.

    Returns:
        Parsed dataframe.
    """
    return pd.read_csv(io.BytesIO(file_bytes))

def make_features_df_from_single(transaction_amount:float, account_age_days:int) -> pd.DataFrame:
    """Create a single-row features dataframe from manual user input.

    Args:
        transaction_amount: Transaction amount value.
        account_age_days: Account age expressed in days.

    Returns:
        Dataframe with exactly the columns expected by `config.FEATURES`.
    """
    df = pd.DataFrame([{
        "Transaction Amount": float(transaction_amount),
        "Account Age Days": int(account_age_days),
    }])
    return df[FEATURES]
def make_features_df_from_batch(df_in: pd.DataFrame) -> pd.DataFrame:
    """Convert a batch input dataframe into model features.

    Expects columns:
    - id
    - amount
    - account_age

    Values are coerced to numeric; rows failing coercion raise an error.

    Args:
        df_in: Raw uploaded dataframe.

    Returns:
        Dataframe containing `config.FEATURES`.

    Raises:
        ValueError: If required columns are missing or conversion to numeric fails.
    """
    required = {"id", "amount", "account_age"}
    missing = required - set(df_in.columns)
    if missing:
        raise ValueError(f"Brakuje kolumn: {missing}, Wymagane: {required}")
    df_feat = pd.DataFrame({
        "Transaction Amount": pd.to_numeric(df_in["amount"], errors="coerce"),
        "Account Age Days": pd.to_numeric(df_in["account_age"], errors="coerce"),
    })
    bad = df_feat.isna().any(axis=1)
    if bool(bad.any()):
        raise ValueError("Nie da sie zrzutować 'ammount'/'account_age' na liczby w niektorych wierszach"
                         f"Indeksy z błędem: {df_in.index[bad].tolist()}")
    return df_feat[FEATURES]
def _is_fraudulent_style(val: bool) -> str:
    """Return CSS styling for the 'Is Fraudulent' column in Streamlit tables.

    Args:
        val: Value of the fraud flag.

    Returns:
        CSS style string for Streamlit's pandas Styler.
    """
    if str(val) == "True":
        return "background-color: #ff4b4b; color: white; font-weight: 700;"
    return "background-color: #2ecc71; color: white; font-weight: 700;"


try:
    bundle = load_model(MODEL_DEFAULT_PATH);
except Exception as e:
    st.error(e)
    st.stop()


mode = st.radio("Wybierz tryb",[
    "Opcja 1: Ręczne dane (Amount + Account Age in Days)",
    "Opcja 2: CSV (id, amount, account_age)",
],horizontal=True)

if mode.startswith("Opcja 1"):
    st.subheader("Opcja 1 - pojedyńcza transakcja")
    c1,c2 = st.columns([1,1])
    with c1:
        amount = st.number_input("Amount",min_value=0, placeholder="Input transaction amount")
    with c2:
        account_age = st.number_input("Account Age (days)",min_value=0, placeholder="Input account age in days")
    if st.button("Sprawdź", type="primary"):
        df_x = make_features_df_from_single(amount, account_age)
        scored = predict_dataframe(bundle, df_x)
        row = scored.iloc[0]

        pred = int(row["predicted_value"])
        p0 = float(row.get("probability_of_value_0", 1.0 - float(row["probability_of_value_1"])))
        p1 = float(row["probability_of_value_1"])
        confidence = max(p0, p1)

        is_fraud = (pred == 1)

        if is_fraud:
            st.error("🚨 Fraudulent: **TAK**")
        else:
            st.success("✅ Fraudulent: **NIE**")

        m2, m3 = st.columns(2)
        m2.metric("Prawdopodobieństwo Fraud", f"{p1:.3f}")
        m3.metric("Pewność (max prob)", f"{confidence:.3f}")

        with st.expander("Szczegóły (raw output)"):
            st.dataframe(scored, width='stretch', hide_index=True)
else:
    st.subheader("Opcja 2: CSV (id, amount, account_age)")
    uploaded = st.file_uploader("Wgraj CSV (kolumny: id, amount, account_age)", type=["csv"])
    if uploaded is None:
        st.info("Wygraj CSV (kolumny: id, amount, account_age)")
        st.stop()
    try:
        df_in = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Nie można wczytać CSV: {e}")
        st.stop()
    st.write("Podgląd danych - 15 wierszy")
    st.dataframe(df_in.head(15), width='stretch', hide_index=True)
    if st.button("Ewaluuj", type="primary"):
        try:
            df_x = make_features_df_from_batch(df_in)
            scored = predict_dataframe(bundle, df_x)

            pred_col = "predicted_value" if "predicted_value" in scored.columns else "prediction"
            pred = scored[pred_col].astype(int)
            out = df_in[["id", "amount", "account_age"]].copy()
            out["Is Fraudulent"] = (pred == 1).map({True: "True", False: "False"})

            styled = out.style.applymap(_is_fraudulent_style, subset=["Is Fraudulent"])

            st.write("Wyniki:")
            st.dataframe(styled, width='stretch', hide_index=True)

        except Exception as e:
            st.error(e)

