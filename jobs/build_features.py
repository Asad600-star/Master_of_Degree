import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# Feature configuration
WINDOWS = [5, 10, 20]
MAX_LAG = 5

# If you want to build for only one symbol, set it here (e.g. "AAPL").
# Leave as None to build for ALL symbols found in market_ohlcv.
ONLY_SYMBOL: str | None = None


def build_features_for_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    df columns expected: date, open, high, low, close, volume (sorted ascending by date)
    """
    df = df.copy()

    # Keep date as "date" (not timestamp) for consistency with market_ohlcv
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Basic returns
    df["return_1d"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"]).diff()

    # Rolling features
    for w in WINDOWS:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
        df[f"volatility_{w}"] = df["return_1d"].rolling(w).std()

    # Lagged returns
    for lag in range(1, MAX_LAG + 1):
        df[f"return_lag_{lag}"] = df["return_1d"].shift(lag)

    # Target: next-day return
    df["target_return_1d"] = df["return_1d"].shift(-1)

    # Add symbol column (IMPORTANT if you build multiple tickers)
    df["symbol"] = symbol

    # Drop rows with NaNs introduced by rolling/lags/target shift
    df = df.dropna().reset_index(drop=True)
    return df


def main() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set. Check your .env file.")

    engine = create_engine(db_url, pool_pre_ping=True)

    # 1) Get list of symbols to build
    if ONLY_SYMBOL:
        symbols = [ONLY_SYMBOL]
    else:
        symbols_df = pd.read_sql_query(
            "SELECT DISTINCT symbol FROM market_ohlcv ORDER BY symbol",
            con=engine,
        )
        symbols = symbols_df["symbol"].tolist()

    if not symbols:
        raise RuntimeError("No symbols found in market_ohlcv. Run ingest_prices.py first.")

    print(f"[INFO] Building features for symbols: {symbols}")

    # 2) Rebuild from scratch (clean table)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS features_daily"))

    total_rows = 0
    created = False

    # 3) Build per symbol and write
    q = text(
        """
        SELECT date, open, high, low, close, volume
        FROM market_ohlcv
        WHERE symbol = :symbol
        ORDER BY date
        """
    )

    for sym in symbols:
        raw = pd.read_sql_query(q, con=engine, params={"symbol": sym})

        if raw.empty:
            print(f"[WARN] No rows in market_ohlcv for symbol={sym}, skipping")
            continue

        feats = build_features_for_symbol(raw, sym)

        if feats.empty:
            print(f"[WARN] Features empty after dropna for symbol={sym}, skipping")
            continue

        # Create table once, then append
        if not created:
            feats.to_sql(
                "features_daily",
                con=engine,
                if_exists="replace",
                index=False,
                method="multi",
                chunksize=2000,
            )
            created = True
        else:
            feats.to_sql(
                "features_daily",
                con=engine,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=2000,
            )

        total_rows += len(feats)
        print(f"[OK] {sym}: features rows={len(feats)}")

    if not created:
        raise RuntimeError("features_daily was not created (no symbols produced features).")

    # 4) Add an index for fast queries (best-effort)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS features_daily_symbol_date_idx "
                "ON features_daily(symbol, date)"
            )
        )

    print(f"[DONE] features_daily built. total_rows={total_rows}")


if __name__ == "__main__":
    main()