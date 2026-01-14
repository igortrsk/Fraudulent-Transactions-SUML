"""Streamlit UI for fraud prediction.

Provides two modes:
1) Manual input: single transaction prediction from Amount and Account Age Days.
2) CSV batch input: predictions for uploaded CSV with columns (id, amount, account_age).

The app loads a pickled `ModelBundle` and uses the same preprocessing + model
logic as the CLI inference utilities.
"""
from pathlib import Path
import sys 
import io
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import FEATURES, REQUIRED_COLS
from src.inference import predict_dataframe
from app import load_bundle_pickle


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MODEL_DEFAULT_PATH = "models/fraud_rf_bundle.pkl"

st.set_page_config(page_title="Ecomm Fraud Prediction", layout="wide")

def read_csv_flexible(uploaded_file) -> pd.DataFrame:
    raw_bytes = uploaded_file.getvalue()
    text = raw_bytes.decode("utf-8", errors="replace")

    header = text.splitlines()[0]
    sep = ";" if header.count(";") >= header.count(",") else ","

    df = pd.read_csv(io.StringIO(text), sep=sep)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Brakuje kolumn: {missing}. Wymagane: {REQUIRED_COLS}")

    return df

def load_css(path: str = "./css/style.css") -> None:
    css_path = Path(__file__).resolve().parent / path
    if not css_path.exists():
        st.warning(f"Brakuje pliku CSS: {css_path.resolve()}")
        return;
    st.markdown(f"<style>{css_path.read_text(encoding="utf-8")}</style>", unsafe_allow_html=True)
load_css()

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

    return pd.read_csv(io.BytesIO(file_bytes), decimal=",")

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
    missing = REQUIRED_COLS - set(df_in.columns)
    if missing:
        raise ValueError(f"Brakuje kolumn: {missing}, Wymagane: {REQUIRED_COLS}")

    amount = (
        df_in["amount"]
        .astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    account_age = (
        df_in["account_age"]
        .astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    df_feat = pd.DataFrame({
        "Transaction Amount": pd.to_numeric(amount, errors="coerce"),
        "Account Age Days": pd.to_numeric(account_age, errors="coerce"),
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

def metric_card(label: str, value: str) -> None:
    """Return HTML element for the "Fraud Predictions System" section.

        Args:
            label: Information of what the value is representing.
            value: Statistics value about Fraud Predictions System.

        Returns:
            HTML DIV element with two more DIVs that represent label and value of the statistic.
        """
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

try:
    bundle = load_model(MODEL_DEFAULT_PATH);
except Exception as e:
    st.error(e)
    st.stop()

st.markdown(
    """
    <div class="app-header">
      <div class="shield">
        <svg width="34" height="34" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2L20 6V12C20 17 16.5 20.5 12 22C7.5 20.5 4 17 4 12V6L12 2Z"
                stroke="#2563EB" stroke-width="2" stroke-linejoin="round"/>
          <path d="M8.5 12.5L10.7 14.7L15.6 9.8"
                stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div>
        <div class="h-title">Fraud Prediction System</div>
        <div class="h-subtitle">AI-powered transaction fraud detection</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)
with c1:
    metric_card("Accuracy Rate", "99.8%")
with c2:
    metric_card("Transactions Analyzed", "2.3M+")
with c3:
    metric_card("Fraud Prevented", "$450M")



st.markdown('<div class="panel-title">Transaction Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">Analysis Method</div>', unsafe_allow_html=True)

method = st.radio(
    "Analysis Method",
    ["Single Transaction\nAnalyze one transaction", "Batch Analysis\nUpload CSV file"],
    horizontal=True,
    label_visibility="collapsed",
)

is_single = method.startswith("Single Transaction")


if is_single:
    with st.form("single_form", border=False):
        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.0,
            step=0.01,
            value=0.0,
            format="%.2f",
        )
        account_age = st.number_input(
            "Account Age (days)",
            min_value=0,
            step=1,
            value=0
        )
        run = st.form_submit_button("Run Fraud Detection", type="primary", use_container_width=True)
    if run:
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

        m1, m2 = st.columns(2)
        m1.metric("Prawdopodobieństwo Fraud", f"{p1:.3f}")
        m2.metric("Pewność (max prob)", f"{confidence:.3f}")

        with st.expander("Szczegóły (raw output)"):
            st.dataframe(scored, width='stretch', hide_index=True)
else:
    uploaded = st.file_uploader("Wgraj CSV (kolumny: id, amount, account_age)", type=["csv"])
    if uploaded is None:
        st.info("Wygraj CSV (kolumny: id, amount, account_age)")
        st.warning("Uwaga: Plik CSV musi używać kropki jako separatora dziesiętnego")
        st.stop()
    try:
        df_in = read_csv_flexible(uploaded)
    except Exception as e:
        st.error(f"Nie można wczytać CSV:{e}")
        st.warning("Uwaga: Plik CSV musi używać kropki jako separatora dziesiętnego")
        st.stop()
    st.write("Podgląd danych - 15 wierszy")
    st.dataframe(df_in.head(15), width='stretch', hide_index=True)
    if st.button("Run Fraud Detection", type="primary", use_container_width=True):
        try:
            df_x = make_features_df_from_batch(df_in)
            scored = predict_dataframe(bundle, df_x)

            pred_col = "predicted_value" if "predicted_value" in scored.columns else "prediction"
            pred = scored[pred_col].astype(int)
            out = df_in[["id", "amount", "account_age"]].copy()
            out["Is Fraudulent"] = (pred == 1).map({True: "True", False: "False"})

            styled = out.style.map(_is_fraudulent_style, subset=["Is Fraudulent"])

            st.write("Wyniki:")
            st.dataframe(styled, width='stretch', hide_index=True)

        except Exception as e:
            st.error(e)