import os
import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Optional Prophet support (install separately)
try:
    from prophet import Prophet  # type: ignore
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def _latest_file(prefix: str) -> str:
    files = [f for f in os.listdir(DATA_DIR) if f.startswith(prefix) and f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No files matching prefix {prefix} in {DATA_DIR}")
    files.sort(reverse=True)
    return os.path.join(DATA_DIR, files[0])


def _features_from_price(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Create simple, robust features: returns, moving averages, volatility.
    """
    out = df.copy()
    out["return_1d"] = out[price_col].pct_change()
    out["ma_5"] = out[price_col].rolling(5).mean()
    out["ma_20"] = out[price_col].rolling(20).mean()
    out["vol_10"] = out[price_col].pct_change().rolling(10).std()
    out["target_next"] = out[price_col].shift(-1)
    out = out.dropna()
    return out


def train_rf_regressor(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=4,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def eval_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    preds = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }


def predict_next(model, last_row: pd.DataFrame) -> float:
    return float(model.predict(last_row)[0])


def run_stock_models() -> Dict[str, Dict]:
    results = {}
    stock_files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("stocks_") and f.endswith(".csv")])
    for fname in stock_files:
        path = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(path)
        df = df.rename(columns={"close": "close"})
        if "close" not in df.columns:
            # yfinance columns are lower-cased by fetcher; ensure 'close' exists
            continue
        df = df[["date", "close", "ticker"]]
        feat = _features_from_price(df, price_col="close")
        if feat.empty:
            continue

        X = feat[["return_1d", "ma_5", "ma_20", "vol_10"]]
        y = feat["target_next"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = train_rf_regressor(X_train, y_train)
        metrics = eval_model(model, X_test, y_test)

        last = X.tail(1)
        next_pred = predict_next(model, last)
        ticker = feat["ticker"].iloc[-1] if "ticker" in feat.columns else fname.split("_")[1]

        results[ticker] = {
            "latest_close": float(feat["close"].iloc[-1]),
            "predicted_next_close": next_pred,
            "metrics": metrics,
            "source_file": fname,
        }
    return results


def run_crypto_models() -> Dict[str, Dict]:
    results = {}
    crypto_files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("crypto_") and f.endswith(".csv")])
    for fname in crypto_files:
        path = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(path)
        if "close" not in df.columns:
            continue
        df = df[["date", "close", "id"]]
        feat = _features_from_price(df, price_col="close")
        if feat.empty:
            continue

        X = feat[["return_1d", "ma_5", "ma_20", "vol_10"]]
        y = feat["target_next"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = train_rf_regressor(X_train, y_train)
        metrics = eval_model(model, X_test, y_test)

        last = X.tail(1)
        next_pred = predict_next(model, last)
        cid = feat["id"].iloc[-1] if "id" in feat.columns else fname.split("_")[1]

        results[cid] = {
            "latest_close": float(feat["close"].iloc[-1]),
            "predicted_next_close": next_pred,
            "metrics": metrics,
            "source_file": fname,
        }
    return results


def save_predictions(stocks: Dict, crypto: Dict) -> Tuple[str, str]:
    stock_path = os.path.join(OUTPUTS_DIR, "predictions_stocks.json")
    crypto_path = os.path.join(OUTPUTS_DIR, "predictions_crypto.json")
    with open(stock_path, "w") as f:
        json.dump(stocks, f, indent=2)
    with open(crypto_path, "w") as f:
        json.dump(crypto, f, indent=2)
    print(f"[info] Saved predictions: {stock_path}, {crypto_path}")
    return stock_path, crypto_path


def save_summary_markdown(stocks: Dict, crypto: Dict) -> str:
    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, "summary.md")
    lines = ["# Daily prediction summary", ""]
    lines.append("## Stocks")
    for k, v in stocks.items():
        lines.append(f"- {k}: latest={v['latest_close']:.4f}, predicted_next={v['predicted_next_close']:.4f}, MAE={v['metrics']['mae']:.4f}, R2={v['metrics']['r2']:.4f}")
    lines.append("")
    lines.append("## Crypto")
    for k, v in crypto.items():
        lines.append(f"- {k}: latest={v['latest_close']:.4f}, predicted_next={v['predicted_next_close']:.4f}, MAE={v['metrics']['mae']:.4f}, R2={v['metrics']['r2']:.4f}")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[info] Saved summary: {path}")
    return path


if __name__ == "__main__":
    stocks = run_stock_models()
    crypto = run_crypto_models()
    save_predictions(stocks, crypto)
    save_summary_markdown(stocks, crypto)
