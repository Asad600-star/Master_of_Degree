"""Latency benchmark for Article 3 (real-time system architecture).

Замеряет время отклика основных компонент системы:
  • db_query_time      — SELECT * FROM predictions_latest WHERE symbol=:s
  • shap_load_time     — чтение SHAP JSON
  • full_pipeline_time — services.predict.get_prediction() полный путь (cached)
  • cold_start_time    — первый запуск get_prediction после рестарта (без кэша)

Выход:
  artifacts/latency_benchmark.csv  — все измерения
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from sqlalchemy import create_engine, text

ART = ROOT / "artifacts"
OUT_CSV = ART / "latency_benchmark.csv"

SYMBOLS = ["AAPL", "TSLA", "^GSPC", "^IXIC"]
N_REPEATS = 50  # повторов на инструмент


def time_it(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - t0) * 1000  # ms
    return elapsed, result


# ─── Bench 1: предсказание из CSV ───
def bench_predictions_csv():
    csv_path = ART / "predictions_latest.csv"
    times = []
    for _ in range(N_REPEATS):
        for sym in SYMBOLS:
            t, _ = time_it(lambda s=sym: pd.read_csv(csv_path)[pd.read_csv(csv_path)["symbol"] == s].iloc[-1])
            times.append({"component": "csv_read_predictions", "symbol": sym, "latency_ms": t})
    return times


# ─── Bench 2: PostgreSQL SELECT ───
def bench_db_query():
    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    # Pre-warm connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1")).fetchall()
    times = []
    q1 = text("SELECT MAX(date) FROM market_ohlcv WHERE symbol = :s")
    q2 = text("SELECT * FROM features_daily WHERE symbol = :s ORDER BY date DESC LIMIT 1")
    for _ in range(N_REPEATS):
        for sym in SYMBOLS:
            # q1 — max date
            with engine.connect() as conn:
                t, _ = time_it(lambda s=sym, c=conn, q=q1: c.execute(q, {"s": s}).fetchone())
                times.append({"component": "db_max_date", "symbol": sym, "latency_ms": t})
            # q2 — latest features
            with engine.connect() as conn:
                t, _ = time_it(lambda s=sym, c=conn, q=q2: c.execute(q, {"s": s}).fetchone())
                times.append({"component": "db_latest_features", "symbol": sym, "latency_ms": t})
    return times


# ─── Bench 3: SHAP JSON load ───
def bench_shap_load():
    times = []
    for _ in range(N_REPEATS):
        for sym in SYMBOLS:
            p = ART / f"shap_{sym}_direction.json"
            if not p.exists(): continue
            t, _ = time_it(lambda pth=p: json.load(open(pth)))
            times.append({"component": "shap_json_load", "symbol": sym, "latency_ms": t})
    return times


# ─── Bench 4: end-to-end services.predict.get_prediction (cached path) ───
def bench_get_prediction():
    from services.predict import get_prediction
    # Warm-up
    for sym in SYMBOLS:
        try:
            get_prediction(sym, refresh=False)
        except Exception:
            pass
    times = []
    for i in range(N_REPEATS):
        for sym in SYMBOLS:
            try:
                t, _ = time_it(get_prediction, sym, refresh=False)
                times.append({"component": "get_prediction_cached", "symbol": sym, "latency_ms": t})
            except Exception as e:
                print(f"  [WARN] {sym}: {e}")
    return times


def main():
    print("[BENCH 1] CSV read of predictions_latest.csv ...")
    csv_times = bench_predictions_csv()
    print(f"  {len(csv_times)} measurements")

    print("[BENCH 2] PostgreSQL queries ...")
    db_times = bench_db_query()
    print(f"  {len(db_times)} measurements")

    print("[BENCH 3] SHAP JSON load ...")
    shap_times = bench_shap_load()
    print(f"  {len(shap_times)} measurements")

    print("[BENCH 4] services.predict.get_prediction ...")
    pred_times = bench_get_prediction()
    print(f"  {len(pred_times)} measurements")

    all_times = csv_times + db_times + shap_times + pred_times
    df = pd.DataFrame(all_times)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] Saved {len(df)} rows to {OUT_CSV}")

    # Quick summary
    print("\n=== Summary by component (mean ± std ms) ===")
    summary = df.groupby("component")["latency_ms"].agg(["mean", "std", "min", "max", "count"]).round(3)
    print(summary)


if __name__ == "__main__":
    main()
