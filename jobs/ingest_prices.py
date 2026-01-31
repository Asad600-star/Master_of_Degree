import os

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


def get_env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name, default)
    if v is None or str(v).strip() == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return str(v).strip()


def parse_symbols(s: str) -> list[str]:
    # allow comma or spaces
    parts = [p.strip() for p in s.replace(" ", ",").split(",")]
    return [p for p in parts if p]


def ensure_market_table(engine) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS market_ohlcv (
        symbol      text NOT NULL,
        date        date NOT NULL,
        open        double precision,
        high        double precision,
        low         double precision,
        close       double precision,
        adj_close   double precision,
        volume      bigint,
        source      text NOT NULL DEFAULT 'yfinance',
        ingested_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (symbol, date)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def download_one(symbol: str, start: str) -> pd.DataFrame:
    df = (
        yf.download(
            symbol,
            start=start,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
        )
        .reset_index()
    )

    if df.empty:
        return df

    # yfinance can return MultiIndex columns sometimes; normalize
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            (c[0] if c[0] else c[1]) if isinstance(c, tuple) else c for c in df.columns
        ]

    rename = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename)

    keep = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    df = df[keep].copy()

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["symbol"] = symbol
    return df


def main() -> None:
    db_url = get_env("DATABASE_URL")
    symbols = parse_symbols(get_env("SYMBOLS"))
    start_date = get_env("START_DATE")
    source = os.environ.get("SOURCE", "yfinance").strip() or "yfinance"

    engine = create_engine(db_url, pool_pre_ping=True)
    ensure_market_table(engine)

    upsert_sql = text(
        """
        INSERT INTO market_ohlcv(symbol, date, open, high, low, close, adj_close, volume, source)
        VALUES (:symbol, :date, :open, :high, :low, :close, :adj_close, :volume, :source)
        ON CONFLICT (symbol, date) DO UPDATE SET
            open        = EXCLUDED.open,
            high        = EXCLUDED.high,
            low         = EXCLUDED.low,
            close       = EXCLUDED.close,
            adj_close   = EXCLUDED.adj_close,
            volume      = EXCLUDED.volume,
            source      = EXCLUDED.source,
            ingested_at = now();
        """
    )

    print(f"[DB] Using DATABASE_URL={db_url}")
    print(f"[INFO] Symbols={symbols}, START_DATE={start_date}, SOURCE={source}")

    total = 0
    for sym in symbols:
        df = download_one(sym, start_date)
        if df.empty:
            print(f"[WARN] {sym}: no data, skipping")
            continue

        rows = df.to_dict(orient="records")
        for r in rows:
            r["source"] = source

        with engine.begin() as conn:
            conn.execute(upsert_sql, rows)

        total += len(df)
        print(f"[OK] {sym}: upserted {len(df)} rows ({df['date'].min()} -> {df['date'].max()})")

    print(f"[DONE] total upserted rows: {total}")


if __name__ == "__main__":
    main()