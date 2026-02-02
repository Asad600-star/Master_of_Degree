import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

WINDOWS = [5, 10, 20]
MAX_LAG = 5
ONLY_SYMBOL: str | None = (os.environ.get("ONLY_SYMBOL") or "").strip() or None


def get_env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name, default)
    if v is None or str(v).strip() == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return str(v).strip()


def build_features_for_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    df["return_1d"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"]).diff()

    for w in WINDOWS:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
        df[f"volatility_{w}"] = df["return_1d"].rolling(w).std()

    for lag in range(1, MAX_LAG + 1):
        df[f"return_lag_{lag}"] = df["return_1d"].shift(lag)

    df["target_return_1d"] = df["return_1d"].shift(-1)
    df["symbol"] = symbol

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df


def ensure_features_table(engine) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS features_daily (
        symbol text,
        date date,
        open double precision,
        high double precision,
        low double precision,
        close double precision,
        volume bigint,
        return_1d double precision,
        log_return double precision,
        sma_5 double precision,
        volatility_5 double precision,
        sma_10 double precision,
        volatility_10 double precision,
        sma_20 double precision,
        volatility_20 double precision,
        return_lag_1 double precision,
        return_lag_2 double precision,
        return_lag_3 double precision,
        return_lag_4 double precision,
        return_lag_5 double precision,
        target_return_1d double precision
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))

        nulls = conn.execute(
            text(
                """
                SELECT
                  SUM((symbol IS NULL)::int) AS symbol_nulls,
                  SUM((date   IS NULL)::int) AS date_nulls
                FROM features_daily
                """
            )
        ).mappings().one()

        if (nulls["symbol_nulls"] or 0) > 0 or (nulls["date_nulls"] or 0) > 0:
            raise RuntimeError(
                "features_daily contains NULLs in key columns: "
                f"symbol_nulls={nulls['symbol_nulls']}, date_nulls={nulls['date_nulls']}. "
                "Fix/delete those rows before enforcing constraints."
            )

        conn.execute(text("ALTER TABLE features_daily ALTER COLUMN symbol SET NOT NULL"))
        conn.execute(text("ALTER TABLE features_daily ALTER COLUMN date SET NOT NULL"))

        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS features_daily_symbol_date_uq "
                "ON features_daily(symbol, date)"
            )
        )
        conn.execute(text("CREATE INDEX IF NOT EXISTS features_daily_date_idx ON features_daily(date)"))
        conn.execute(text("DROP INDEX IF EXISTS features_daily_symbol_date_idx"))


def main() -> None:
    db_url = get_env("DATABASE_URL")
    engine = create_engine(db_url, pool_pre_ping=True)
    ensure_features_table(engine)

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

    q = text(
        """
        SELECT date, open, high, low, close, volume
        FROM market_ohlcv
        WHERE symbol = :symbol
        ORDER BY date
        """
    )

    cols = [
        "symbol","date","open","high","low","close","volume",
        "return_1d","log_return",
        "sma_5","volatility_5",
        "sma_10","volatility_10",
        "sma_20","volatility_20",
        "return_lag_1","return_lag_2","return_lag_3","return_lag_4","return_lag_5",
        "target_return_1d",
    ]

    upsert_sql = text(
        """
        INSERT INTO features_daily(
            symbol, date, open, high, low, close, volume,
            return_1d, log_return,
            sma_5, volatility_5,
            sma_10, volatility_10,
            sma_20, volatility_20,
            return_lag_1, return_lag_2, return_lag_3, return_lag_4, return_lag_5,
            target_return_1d
        )
        VALUES (
            :symbol, :date, :open, :high, :low, :close, :volume,
            :return_1d, :log_return,
            :sma_5, :volatility_5,
            :sma_10, :volatility_10,
            :sma_20, :volatility_20,
            :return_lag_1, :return_lag_2, :return_lag_3, :return_lag_4, :return_lag_5,
            :target_return_1d
        )
        ON CONFLICT (symbol, date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            return_1d = EXCLUDED.return_1d,
            log_return = EXCLUDED.log_return,
            sma_5 = EXCLUDED.sma_5,
            volatility_5 = EXCLUDED.volatility_5,
            sma_10 = EXCLUDED.sma_10,
            volatility_10 = EXCLUDED.volatility_10,
            sma_20 = EXCLUDED.sma_20,
            volatility_20 = EXCLUDED.volatility_20,
            return_lag_1 = EXCLUDED.return_lag_1,
            return_lag_2 = EXCLUDED.return_lag_2,
            return_lag_3 = EXCLUDED.return_lag_3,
            return_lag_4 = EXCLUDED.return_lag_4,
            return_lag_5 = EXCLUDED.return_lag_5,
            target_return_1d = EXCLUDED.target_return_1d;
        """
    )

    total_rows = 0
    for sym in symbols:
        raw = pd.read_sql_query(q, con=engine, params={"symbol": sym})
        if raw.empty:
            print(f"[WARN] No rows in market_ohlcv for symbol={sym}, skipping")
            continue

        feats = build_features_for_symbol(raw, sym)
        if feats.empty:
            print(f"[WARN] Features empty after cleaning for symbol={sym}, skipping")
            continue

        feats = feats[cols]
        rows = feats.to_dict(orient="records")

        with engine.begin() as conn:
            conn.execute(upsert_sql, rows)

        total_rows += len(feats)
        print(f"[OK] {sym}: features rows={len(feats)}")

    print(f"[DONE] features_daily built. total_rows={total_rows}")


if __name__ == "__main__":
    main()