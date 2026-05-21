"""ExtraTrees ablation study for realised-volatility forecasting.

Cвипы:
  • n_estimators ∈ {100, 250, 500, 1000}      (фиксируем max_depth=12, min_samples_leaf=10)
  • max_depth   ∈ {6, 9, 12, 15, None}        (фиксируем n_estimators=500, min_samples_leaf=10)
  • min_samples_leaf ∈ {5, 10, 20, 50}        (фиксируем n_estimators=500, max_depth=12)

Протокол: тот же walk-forward (W_train=1200, W_v=W_t=126, step=63), что и в
jobs/train_baseline.py — для совместимости с уже измеренной baseline-точкой.

Запись: artifacts/et_ablation_volatility_k5.csv  с колонками
  symbol, fold, hyperparam_kind, n_estimators, max_depth, min_samples_leaf,
  rmse, mae, r2, train_seconds, predict_seconds, n_train, n_test
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from sqlalchemy import create_engine, text

# ─── Парам walk-forward (совпадают с jobs/train_baseline.py) ───
W_TRAIN = 1200
W_VAL = 126
W_TEST = 126
STEP = 63
HORIZON = 5
RANDOM_STATE = 42

# ─── Сетка гиперпараметров ───
ABL_N = [100, 250, 500, 1000]
ABL_DEPTH = [6, 9, 12, 15, None]
ABL_LEAF = [5, 10, 20, 50]

SYMBOLS = ["AAPL", "TSLA", "^GSPC", "^IXIC"]

OUT_CSV = ROOT / "artifacts" / "et_ablation_volatility_k5.csv"


# ─── Загрузка признаков (тот же набор что в train_baseline.py) ───
def load_features_for_symbol(engine, symbol: str) -> pd.DataFrame:
    q = text("SELECT * FROM features_daily WHERE symbol = :s ORDER BY date")
    df = pd.read_sql_query(q, con=engine, params={"s": symbol}, parse_dates=["date"])
    return df


FEATURE_COLS = [
    "open", "high", "low", "close", "volume", "return_1d", "log_return",
    "sma_5", "volatility_5", "sma_10", "volatility_10", "sma_20", "volatility_20",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_4", "return_lag_5",
    "mkt_return_1d", "mkt_log_return", "mkt_mom_5", "mkt_mom_10", "mkt_mom_20",
    "mkt_vol_20", "vix_level", "vix_return_1d", "vix_change_1d",
    "irx_level", "irx_change_1d", "tnx_level", "tnx_change_1d",
]


def compute_realized_volatility(df: pd.DataFrame, k: int) -> pd.Series:
    """target_vol_kd = std(log_return[t+1..t+k], ddof=0)."""
    fut = [df["log_return"].shift(-i) for i in range(1, k + 1)]
    tmp = pd.concat(fut, axis=1)
    return tmp.std(axis=1, ddof=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2}


def run_one_config(X_train, y_train, X_test, y_test, *, n_estimators, max_depth, min_samples_leaf):
    """Обучить и оценить одну конфигурацию ExtraTreesRegressor."""
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    t_train = time.perf_counter() - t0
    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    t_predict = time.perf_counter() - t1
    metrics = compute_metrics(y_test, y_pred)
    return metrics, t_train, t_predict


def grid() -> Iterable[tuple[str, dict]]:
    """Сетка экспериментов — три «оси» относительно baseline (N=500, depth=12, leaf=10)."""
    baseline = {"n_estimators": 500, "max_depth": 12, "min_samples_leaf": 10}
    seen = set()

    def emit(kind, params):
        key = (kind, params["n_estimators"], params["max_depth"], params["min_samples_leaf"])
        if key in seen:
            return
        seen.add(key)
        return kind, params

    # ось 1: n_estimators
    for n in ABL_N:
        r = emit("n_estimators", {**baseline, "n_estimators": n})
        if r:
            yield r
    # ось 2: max_depth
    for d in ABL_DEPTH:
        r = emit("max_depth", {**baseline, "max_depth": d})
        if r:
            yield r
    # ось 3: min_samples_leaf
    for l in ABL_LEAF:
        r = emit("min_samples_leaf", {**baseline, "min_samples_leaf": l})
        if r:
            yield r


def main():
    db_url = os.environ["DATABASE_URL"]
    engine = create_engine(db_url, pool_pre_ping=True)

    print(f"[INFO] Walk-forward params: W_train={W_TRAIN}, W_v={W_VAL}, W_t={W_TEST}, step={STEP}, horizon={HORIZON}")
    print(f"[INFO] Symbols: {SYMBOLS}")
    print(f"[INFO] Ablation grid: {len(list(grid()))} unique configurations")

    rows = []
    total_runs = 0

    for symbol in SYMBOLS:
        print(f"\n[INFO] Loading features for {symbol} ...")
        df = load_features_for_symbol(engine, symbol)
        df = df.sort_values("date").reset_index(drop=True)
        df["target_vol_kd"] = compute_realized_volatility(df, HORIZON)

        # Drop rows with NaN in needed columns
        cols = FEATURE_COLS + ["target_vol_kd"]
        df = df[cols].dropna().reset_index(drop=True)
        print(f"   {symbol}: {len(df)} rows usable")

        if len(df) < W_TRAIN + W_VAL + W_TEST:
            print(f"   [WARN] not enough rows for {symbol}, skipping")
            continue

        # Walk-forward folds
        fold = 0
        i = W_TRAIN
        while i + W_VAL + W_TEST <= len(df):
            X_train = df.iloc[:i][FEATURE_COLS].values
            y_train = df.iloc[:i]["target_vol_kd"].values
            # We use VAL+TEST combined as the held-out evaluation slot to match earlier protocol
            # (in train_baseline.py, val and test are reported separately — here we use test only)
            X_test = df.iloc[i + W_VAL : i + W_VAL + W_TEST][FEATURE_COLS].values
            y_test = df.iloc[i + W_VAL : i + W_VAL + W_TEST]["target_vol_kd"].values

            for kind, params in grid():
                try:
                    metrics, t_train, t_predict = run_one_config(
                        X_train, y_train, X_test, y_test, **params
                    )
                    rows.append({
                        "symbol": symbol,
                        "fold": fold,
                        "hyperparam_kind": kind,
                        "n_estimators": params["n_estimators"],
                        "max_depth": str(params["max_depth"]),
                        "min_samples_leaf": params["min_samples_leaf"],
                        **metrics,
                        "train_seconds": round(t_train, 4),
                        "predict_seconds": round(t_predict, 4),
                        "n_train": len(X_train),
                        "n_test": len(X_test),
                    })
                    total_runs += 1
                except Exception as e:
                    print(f"   [WARN] {symbol} fold={fold} {kind}/{params}: {e}")

            print(f"   {symbol} fold={fold} done")
            fold += 1
            i += STEP

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] Saved {len(df_out)} rows to {OUT_CSV}")
    print(f"[INFO] Total successful runs: {total_runs}")


if __name__ == "__main__":
    main()
