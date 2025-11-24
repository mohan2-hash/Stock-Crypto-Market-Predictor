import os

from data_fetcher import fetch_stock_history, fetch_crypto_history, save_manifest
from predictor import run_stock_models, run_crypto_models, save_predictions, save_summary_markdown
from visualizer import generate_all_charts

DEFAULT_STOCK_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
DEFAULT_CRYPTO_IDS = ["bitcoin", "ethereum", "solana", "ripple", "cardano"]


def run_pipeline():
    print("[stage] Fetching data")
    stocks = fetch_stock_history(DEFAULT_STOCK_TICKERS)
    crypto = fetch_crypto_history(DEFAULT_CRYPTO_IDS)
    save_manifest({"stocks": stocks, "crypto": crypto})

    print("[stage] Modeling predictions")
    stock_preds = run_stock_models()
    crypto_preds = run_crypto_models()
    save_predictions(stock_preds, crypto_preds)
    save_summary_markdown(stock_preds, crypto_preds)

    print("[stage] Generating charts")
    generate_all_charts()

    print("[done] Pipeline completed")


if __name__ == "__main__":
    # Change working dir to src for imports when executed from root
    os.chdir(os.path.dirname(__file__))
    run_pipeline()
