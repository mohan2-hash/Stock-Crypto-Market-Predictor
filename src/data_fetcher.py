import os
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd
import requests
import yfinance as yf

# =========================
# CONFIG
# =========================
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEFAULT_STOCK_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
DEFAULT_CRYPTO_IDS = ["bitcoin", "ethereum", "solana", "ripple", "cardano"]
LOOKBACK_DAYS = 365  # fetch 1 year history
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

os.makedirs(DATA_DIR, exist_ok=True)


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")


def fetch_stock_history(tickers: List[str], lookback_days: int = LOOKBACK_DAYS) -> Dict[str, str]:
    """
    Fetch historical OHLCV for given stock tickers using yfinance.
    Saves one CSV per ticker.
    Returns dict mapping ticker -> saved filepath.
    """
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    saved = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start.date(), end=end.date(), interval="1d", auto_adjust=True, progress=False)
            if df.empty:
                print(f"[warn] No data for {t}")
                continue
            df = df.reset_index().rename(columns=str.lower)
            df["ticker"] = t
            path = os.path.join(DATA_DIR, f"stocks_{t}_{_timestamp()}.csv")
            df.to_csv(path, index=False)
            saved[t] = path
            print(f"[info] Saved stock data: {path}")
            time.sleep(0.3)  # polite sleep
        except Exception as e:
            print(f"[error] Stock fetch failed for {t}: {e}")
    return saved


def fetch_crypto_history(ids: List[str], lookback_days: int = LOOKBACK_DAYS) -> Dict[str, str]:
    """
    Fetch daily OHLC from CoinGecko /coins/{id}/market_chart endpoint.
    Saves one CSV per crypto id.
    """
    saved = {}
    for cid in ids:
        try:
            days = lookback_days
            url = f"{COINGECKO_BASE}/coins/{cid}/market_chart"
            params = {"vs_currency": "usd", "days": days, "interval": "daily"}
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            # CoinGecko returns prices, market_caps, total_volumes as [timestamp(ms), value]
            prices = pd.DataFrame(data.get("prices", []), columns=["timestamp_ms", "close"])
            market_caps = pd.DataFrame(data.get("market_caps", []), columns=["timestamp_ms", "market_cap"])
            volumes = pd.DataFrame(data.get("total_volumes", []), columns=["timestamp_ms", "volume"])

            df = prices.merge(market_caps, on="timestamp_ms", how="left").merge(volumes, on="timestamp_ms", how="left")
            df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.date
            df["id"] = cid
            df = df[["date", "close", "market_cap", "volume", "id"]].sort_values("date")

            path = os.path.join(DATA_DIR, f"crypto_{cid}_{_timestamp()}.csv")
            df.to_csv(path, index=False)
            saved[cid] = path
            print(f"[info] Saved crypto data: {path}")
            time.sleep(0.3)
        except Exception as e:
            print(f"[error] Crypto fetch failed for {cid}: {e}")
    return saved


def save_manifest(manifest: Dict[str, Dict[str, str]]) -> str:
    """
    Save a manifest JSON with the latest files for stocks and crypto.
    """
    path = os.path.join(DATA_DIR, f"manifest_{_timestamp()}.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[info] Saved manifest: {path}")
    return path


if __name__ == "__main__":
    stocks = fetch_stock_history(DEFAULT_STOCK_TICKERS)
    crypto = fetch_crypto_history(DEFAULT_CRYPTO_IDS)
    save_manifest({"stocks": stocks, "crypto": crypto})
