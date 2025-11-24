import os
from typing import Dict

import pandas as pd
import plotly.graph_objects as go

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def _latest_file(prefix: str):
    files = [f for f in os.listdir(DATA_DIR) if f.startswith(prefix) and f.endswith(".csv")]
    if not files:
        return None
    files.sort(reverse=True)
    return os.path.join(DATA_DIR, files[0])


def plot_price_series(csv_path: str, label: str, price_col: str = "close") -> str:
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        x = df["date"]
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        x = df["datetime"]
    else:
        x = list(range(len(df)))
    y = df[price_col]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label))
    fig.update_layout(title=f"{label} - Price", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
    outpath = os.path.join(REPORTS_DIR, f"chart_{label}.png")
    fig.write_image(outpath, scale=2)
    print(f"[info] Saved chart: {outpath}")
    return outpath


def generate_all_charts() -> Dict[str, str]:
    paths = {}
    stock_file = _latest_file("stocks_")
    crypto_file = _latest_file("crypto_")
    if stock_file:
        label = os.path.basename(stock_file).split("_")[1]
        paths[f"stock_{label}"] = plot_price_series(stock_file, f"Stock {label}")
    if crypto_file:
        label = os.path.basename(crypto_file).split("_")[1]
        paths[f"crypto_{label}"] = plot_price_series(crypto_file, f"Crypto {label}")
    return paths


if __name__ == "__main__":
    generate_all_charts()
