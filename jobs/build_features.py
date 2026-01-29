import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

SYMBOL = "AAPL"
WINDOWS = [5, 10, 20]


def main() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set. Check your .env file.")

    engine = create_engine(db_url, pool_pre_ping=True)

    query = text("""
        SELECT date, open, high, low, close, volume
        FROM market_ohlcv
        WHERE symbol = :symbol
        ORDER BY date
    """)

    df = pd.read_sql_query(query, con=engine, params={"symbol": SYMBOL})

    if df.empty:
        raise RuntimeError(f"No rows found in market_ohlcv for symbol={SYMBOL}")

    df["date"] = pd.to_datetime(df["date"])

    df["return_1d"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"]).diff()

    for w in WINDOWS:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
        df[f"volatility_{w}"] = df["return_1d"].rolling(w).std()

    for lag in range(1, 6):
        df[f"return_lag_{lag}"] = df["return_1d"].shift(lag)

    df["target_return_1d"] = df["return_1d"].shift(-1)

    df = df.dropna().reset_index(drop=True)

    df.to_sql(
        "features_daily",
        con=engine,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=2000,
    )

    print(f"[OK] features_daily saved, rows={len(df)}")


if __name__ == "__main__":
    main()