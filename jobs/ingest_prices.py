import os
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

load_dotenv()

TICKERS = ["AAPL", "TSLA", "^GSPC", "^IXIC"]
START = "2015-01-01"
INTERVAL = "1d"


def download_prices(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=START,
        interval=INTERVAL,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    # yfinance может вернуть MultiIndex колонки даже для одного тикера
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # Date / Datetime могут называться по-разному
    if "Date" in df.columns:
        base_col = "Date"
    elif "Datetime" in df.columns:
        base_col = "Datetime"
    elif "index" in df.columns:
        base_col = "index"
    else:
        raise RuntimeError(f"Unexpected columns from yfinance for {ticker}: {list(df.columns)}")

    df["date"] = pd.to_datetime(df[base_col]).dt.date

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    keep = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns for {ticker}: {missing}. Got: {list(df.columns)}")

    df = df[keep]
    df["symbol"] = ticker
    df["source"] = "yfinance"
    return df


def upsert_prices(engine, df: pd.DataFrame) -> int:
    sql = text(
        """
        INSERT INTO market_ohlcv(symbol, date, open, high, low, close, adj_close, volume, source)
        VALUES (:symbol, :date, :open, :high, :low, :close, :adj_close, :volume, :source)
        ON CONFLICT (symbol, date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low  = EXCLUDED.low,
            close = EXCLUDED.close,
            adj_close = EXCLUDED.adj_close,
            volume = EXCLUDED.volume,
            source = EXCLUDED.source,
            ingested_at = now();
    """
    )
    rows = df.to_dict(orient="records")
    try:
        with engine.begin() as conn:
            conn.execute(sql, rows)
    except SQLAlchemyError as e:
        sample_keys = list(rows[0].keys()) if rows else []
        raise RuntimeError(f"Upsert failed: {e}. Sample row keys: {sample_keys}")
    return len(rows)


def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set. Check your .env file.")

    print(f"[DB] Using DATABASE_URL={db_url}")
    engine = create_engine(db_url, pool_pre_ping=True)

    # проверка подключения
    with engine.connect() as conn:
        v = conn.execute(text("SELECT version()"))
        print("[DB]", v.scalar())

    total = 0
    for t in TICKERS:
        df = download_prices(t)
        inserted = upsert_prices(engine, df)
        total += inserted
        print(f"[OK] {t}: upserted {inserted} rows ({df['date'].min()} -> {df['date'].max()})")

    print(f"[DONE] total upserted rows: {total}")


if __name__ == "__main__":
    main()