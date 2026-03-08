import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
PREDICTIONS_FILE = ARTIFACTS_DIR / "predictions_latest.csv"


def _run(cmd: list[str], extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _refresh_pipeline(symbol: str, horizon: int) -> None:
    _run([sys.executable, "jobs/ingest_prices.py"])
    _run([sys.executable, "jobs/build_features.py"])
    _run(
        [sys.executable, "jobs/train_baseline.py"],
        {
            "ACTION": "infer",
            "ONLY_SYMBOL": symbol,
            "HORIZON_DAYS": str(horizon),
        },
    )


def predict(symbol: str, horizon: int = 5, refresh: bool = False) -> dict:
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise ValueError("symbol is required")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    if refresh or (not PREDICTIONS_FILE.exists()):
        _refresh_pipeline(symbol=symbol, horizon=horizon)

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(f"Predictions file not found: {PREDICTIONS_FILE}")

    df = pd.read_csv(PREDICTIONS_FILE)
    if df.empty:
        raise RuntimeError("predictions_latest.csv is empty")

    if "symbol" not in df.columns:
        raise RuntimeError("predictions_latest.csv does not contain 'symbol' column")

    row = df.loc[df["symbol"].astype(str).str.upper() == symbol]

    if row.empty and not refresh:
        _refresh_pipeline(symbol=symbol, horizon=horizon)

        if not PREDICTIONS_FILE.exists():
            raise FileNotFoundError(f"Predictions file not found after refresh: {PREDICTIONS_FILE}")

        df = pd.read_csv(PREDICTIONS_FILE)
        if df.empty:
            raise RuntimeError("predictions_latest.csv is empty after refresh")

        if "symbol" not in df.columns:
            raise RuntimeError("predictions_latest.csv does not contain 'symbol' column after refresh")

        row = df.loc[df["symbol"].astype(str).str.upper() == symbol]

    if row.empty:
        raise ValueError(f"No prediction found for symbol={symbol}")
    
    rec = row.iloc[-1].to_dict()
    rec["horizon_days"] = int(rec["horizon_days"])
    rec["p_up"] = float(rec["p_up"])
    rec["vol_pred"] = float(rec["vol_pred"])
    return rec