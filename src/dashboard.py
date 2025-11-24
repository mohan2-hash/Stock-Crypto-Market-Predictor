import os
import json

import pandas as pd
import streamlit as st
import plotly.express as px

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

st.set_page_config(page_title="Stock & Crypto Market Predictor", layout="wide")

st.title("📈 Stock & Crypto Market Predictor")

# Predictions
stocks_path = os.path.join(OUTPUTS_DIR, "predictions_stocks.json")
crypto_path = os.path.join(OUTPUTS_DIR, "predictions_crypto.json")

st.sidebar.header("Settings")
asset_type = st.sidebar.selectbox("Asset type", ["Stocks", "Crypto"])

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)

stocks_pred = load_json(stocks_path)
crypto_pred = load_json(crypto_path)

if asset_type == "Stocks":
    st.subheader("Stocks predictions")
    if stocks_pred:
        df = pd.DataFrame.from_dict(stocks_pred, orient="index")
        metrics = df["metrics"].apply(pd.Series)
        df = pd.concat([df.drop(columns=["metrics"]), metrics], axis=1)
        st.dataframe(df)
        pick = st.selectbox("Select ticker", list(stocks_pred.keys()))
        st.metric("Latest close", f"{stocks_pred[pick]['latest_close']:.4f}")
        st.metric("Predicted next close", f"{stocks_pred[pick]['predicted_next_close']:.4f}")
        st.metric("MAE", f"{stocks_pred[pick]['metrics']['mae']:.4f}")
        st.metric("R2", f"{stocks_pred[pick]['metrics']['r2']:.4f}")
    else:
        st.info("No stock predictions found. Run main.py to generate.")
else:
    st.subheader("Crypto predictions")
    if crypto_pred:
        df = pd.DataFrame.from_dict(crypto_pred, orient="index")
        metrics = df["metrics"].apply(pd.Series)
        df = pd.concat([df.drop(columns=["metrics"]), metrics], axis=1)
        st.dataframe(df)
        pick = st.selectbox("Select coin", list(crypto_pred.keys()))
        st.metric("Latest close", f"{crypto_pred[pick]['latest_close']:.4f}")
        st.metric("Predicted next close", f"{crypto_pred[pick]['predicted_next_close']:.4f}")
        st.metric("MAE", f"{crypto_pred[pick]['metrics']['mae']:.4f}")
        st.metric("R2", f"{crypto_pred[pick]['metrics']['r2']:.4f}")
    else:
        st.info("No crypto predictions found. Run main.py to generate.")

st.markdown("---")
st.caption("This dashboard is for educational and demonstration purposes; not financial advice.")
