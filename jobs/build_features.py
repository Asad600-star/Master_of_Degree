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

    # 2) Ensure destination table exists (idempotent) and has a proper uniqueness constraint
    # IMPORTANT: `CREATE TABLE IF NOT EXISTS ... PRIMARY KEY ...` does NOT retrofit an existing table.
    # So we:
    #   - create the table if missing
    #   - ensure symbol/date are NOT NULL
    #   - ensure there is a UNIQUE index on (symbol, date) so ON CONFLICT (symbol, date) works
    ddl_create = """
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
        # Create table if missing
        conn.execute(text(ddl_create))

        # If there are NULLs in symbol/date, we must stop (otherwise NOT NULL/UNIQUE enforcement can fail)
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
                f"features_daily contains NULLs (symbol_nulls={nulls['symbol_nulls']}, date_nulls={nulls['date_nulls']}). "
                "Fix the data or rebuild the table before enabling constraints."
            )

        # Enforce NOT NULL for conflict keys
        conn.execute(text("ALTER TABLE features_daily ALTER COLUMN symbol SET NOT NULL"))
        conn.execute(text("ALTER TABLE features_daily ALTER COLUMN date SET NOT NULL"))

        # Ensure a UNIQUE index exists so ON CONFLICT (symbol, date) is valid
        # (Do not rely on a non-unique btree index.)
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS features_daily_symbol_date_uq "
                "ON features_daily(symbol, date)"
            )
        )

        # Helpful index for time-slicing across all symbols
        conn.execute(text("CREATE INDEX IF NOT EXISTS features_daily_date_idx ON features_daily(date)"))

    total_rows = 0

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

        # Safety: convert +/-inf to NaN then drop missing rows
        feats = feats.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

        if feats.empty:
            print(f"[WARN] Features empty after dropna/inf-clean for symbol={sym}, skipping")
            continue

        # Upsert into features_daily (no duplicates, idempotent)
        cols = [
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "return_1d",
            "log_return",
            "sma_5",
            "volatility_5",
            "sma_10",
            "volatility_10",
            "sma_20",
            "volatility_20",
            "return_lag_1",
            "return_lag_2",
            "return_lag_3",
            "return_lag_4",
            "return_lag_5",
            "target_return_1d",
        ]

        feats = feats[cols]
        rows = feats.to_dict(orient="records")

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

        with engine.begin() as conn:
            conn.execute(upsert_sql, rows)

        total_rows += len(feats)
        print(f"[OK] {sym}: features rows={len(feats)}")

    # 4) Ensure composite index exists (best-effort)
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