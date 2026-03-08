import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
PREDICTIONS_FILE = ARTIFACTS_DIR / "predictions_latest.csv"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL_NAME_MAP = {
    "AAPL": "Apple Inc.",
    "TSLA": "Tesla Inc.",
    "^GSPC": "S&P 500 Index",
    "^IXIC": "Nasdaq Composite",
}

VALID_SYMBOLS = set(SYMBOL_NAME_MAP.keys())


class PredictionServiceError(RuntimeError):
    pass


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run(cmd: list[str], extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _normalize_symbol(symbol: str) -> str:
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise ValueError("symbol is required")
    if symbol not in VALID_SYMBOLS:
        raise ValueError(f"Unsupported symbol: {symbol}. Allowed: {sorted(VALID_SYMBOLS)}")
    return symbol


def refresh_data(symbol: str | None = None) -> dict:
    extra_env: dict[str, str] = {}
    if symbol:
        extra_env["ONLY_SYMBOL"] = _normalize_symbol(symbol)

    started_at = _utc_now_iso()
    _run([sys.executable, "jobs/ingest_prices.py"])
    _run([sys.executable, "jobs/build_features.py"], extra_env=extra_env)
    finished_at = _utc_now_iso()

    return {
        "status": "ok",
        "action": "refresh",
        "symbol": symbol,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
    }


def retrain_models(horizon: int = 5, symbol: str | None = None) -> dict:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    extra_env: dict[str, str] = {
        "HORIZON_DAYS": str(horizon),
        "MODE": "walk",
    }
    if symbol:
        extra_env["ONLY_SYMBOL"] = _normalize_symbol(symbol)

    started_at = _utc_now_iso()
    _run([sys.executable, "jobs/train_baseline.py"], {**extra_env, "TASK": "direction"})
    _run([sys.executable, "jobs/train_baseline.py"], {**extra_env, "TASK": "volatility"})
    finished_at = _utc_now_iso()

    return {
        "status": "ok",
        "action": "retrain",
        "symbol": symbol,
        "horizon_days": horizon,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
    }


def run_inference(horizon: int = 5, symbol: str | None = None) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    extra_env: dict[str, str] = {
        "ACTION": "infer",
        "HORIZON_DAYS": str(horizon),
    }
    if symbol:
        extra_env["ONLY_SYMBOL"] = _normalize_symbol(symbol)

    _run([sys.executable, "jobs/train_baseline.py"], extra_env)

    if not PREDICTIONS_FILE.exists():
        raise PredictionServiceError(f"Predictions file not found: {PREDICTIONS_FILE}")

    df = pd.read_csv(PREDICTIONS_FILE)
    if df.empty:
        raise PredictionServiceError("predictions_latest.csv is empty")

    return df


def load_model_registry(horizon: int = 5) -> pd.DataFrame:
    path = ARTIFACTS_DIR / f"model_registry_k{horizon}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["task", "symbol", "model", "metric_name", "metric_value", "selection_note"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["task", "symbol", "model", "metric_name", "metric_value", "selection_note"])
    return df


def _metric_or_none(value):
    try:
        return float(value)
    except Exception:
        return None


def interpret_signal(pred: dict, registry: pd.DataFrame | None = None) -> dict:
    symbol = str(pred["symbol"])
    p_up = float(pred["p_up"])
    vol_pred = float(pred["vol_pred"])

    if p_up >= 0.60:
        direction_bias = "bullish"
        action = "long bias"
    elif p_up <= 0.40:
        direction_bias = "bearish"
        action = "defensive / no long"
    else:
        direction_bias = "neutral"
        action = "wait / weak edge"

    confidence = abs(p_up - 0.50) * 2.0
    if confidence >= 0.40:
        confidence_label = "high"
    elif confidence >= 0.20:
        confidence_label = "medium"
    else:
        confidence_label = "low"

    if vol_pred >= 0.03:
        risk_label = "high"
    elif vol_pred >= 0.015:
        risk_label = "moderate"
    else:
        risk_label = "low"

    model_meta: dict[str, object] = {}
    if registry is not None and not registry.empty:
        sub = registry[registry["symbol"].astype(str) == symbol].copy()
        if not sub.empty:
            for task_name in ["direction", "volatility"]:
                row = sub[sub["task"].astype(str) == task_name]
                if not row.empty:
                    rec = row.iloc[-1].to_dict()
                    model_meta[f"{task_name}_metric_name"] = rec.get("metric_name")
                    model_meta[f"{task_name}_metric_value"] = _metric_or_none(rec.get("metric_value"))

    return {
        "instrument_name": SYMBOL_NAME_MAP.get(symbol, symbol),
        "direction_bias": direction_bias,
        "confidence_label": confidence_label,
        "risk_label": risk_label,
        "signal_action": action,
        "user_summary": (
            f"{SYMBOL_NAME_MAP.get(symbol, symbol)}: {direction_bias} bias, "
            f"confidence={confidence_label}, risk={risk_label}, "
            f"p_up={p_up:.3f}, expected_vol={vol_pred:.4f}."
        ),
        **model_meta,
    }


def predict(symbol: str, horizon: int = 5, refresh: bool = False) -> dict:
    symbol = _normalize_symbol(symbol)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    refresh_info = None
    if refresh:
        refresh_info = refresh_data(symbol=symbol)

    if refresh or (not PREDICTIONS_FILE.exists()):
        run_inference(horizon=horizon, symbol=symbol)

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(f"Predictions file not found: {PREDICTIONS_FILE}")

    df = pd.read_csv(PREDICTIONS_FILE)
    if df.empty:
        raise PredictionServiceError("predictions_latest.csv is empty")

    if "symbol" not in df.columns:
        raise PredictionServiceError("predictions_latest.csv does not contain 'symbol' column")

    row = df.loc[df["symbol"].astype(str).str.upper() == symbol]

    if row.empty:
        run_inference(horizon=horizon, symbol=symbol)
        df = pd.read_csv(PREDICTIONS_FILE)
        row = df.loc[df["symbol"].astype(str).str.upper() == symbol]

    if row.empty:
        raise ValueError(f"No prediction found for symbol={symbol}")

    rec = row.iloc[-1].to_dict()
    rec["horizon_days"] = int(rec["horizon_days"])
    rec["p_up"] = float(rec["p_up"])
    rec["vol_pred"] = float(rec["vol_pred"])
    rec["generated_at_utc"] = _utc_now_iso()

    registry = load_model_registry(horizon=horizon)
    rec.update(interpret_signal(rec, registry=registry))

    if refresh_info is not None:
        rec["refresh_info"] = refresh_info

    return rec