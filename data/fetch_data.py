import yfinance as yf
import pandas as pd
import argparse
import os

def get_data(
    symbol="AAPL", 
    start="2021-01-01", 
    end="2024-12-31", 
    interval="1d",
    assets=None,
    preprocess=True,
    save=True,
    indicators_fn=None
):
    """
    Download and preprocess historical data for one or more assets.

    Args:
        symbol (str): Default ticker (if assets is None)
        start (str): Start date
        end (str): End date
        interval (str): Data interval (e.g., '1d', '1h', etc.)
        assets (list): List of tickers, if multi-asset
        preprocess (bool): Add indicators (if True)
        save (bool): Save CSV (if True)
        indicators_fn (callable): Function to add indicators (optional)
    Returns:
        dict of pd.DataFrame if multi-asset, else pd.DataFrame
    """
    os.makedirs("data", exist_ok=True)
    result = {}

    tickers = assets if assets else [symbol]
    for sym in tickers:
        print(f"Fetching {sym} ({interval})...")
        df = yf.download(sym, start=start, end=end, interval=interval, progress=False)
        if df.empty or not set(["Open", "High", "Low", "Close", "Volume"]).issubset(df.columns):
            print(f"Warning: {sym} returned no data.")
            continue
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.dropna(inplace=True)
        if preprocess and indicators_fn:
            df = indicators_fn(df)
        if save:
            fname = f"data/{sym.replace('-', '_')}_{interval}.csv"
            df.to_csv(fname)
            print(f"Saved: {fname}")
        result[sym] = df

    if assets:
        return result
    return result[symbol]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="AAPL", help="Ticker symbol (e.g. AAPL, BTC-USD)")
    parser.add_argument("--start", type=str, default="2021-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--multi", nargs='+', default=None, help="List of tickers for multi-asset mode")
    args = parser.parse_args()

    # Optional: Lazy import for indicators if available
    try:
        from utils.indicators import add_indicators
    except ImportError:
        add_indicators = None

    get_data(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        interval=args.interval,
        assets=args.multi,
        preprocess=bool(add_indicators),
        indicators_fn=add_indicators
    )
