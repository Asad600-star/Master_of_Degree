import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import json

from core.risk.risk_manager import RiskManager

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
PREDICTIONS_FILE = ARTIFACTS_DIR / "predictions_latest.csv"

risk_manager = RiskManager()

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _run(cmd: list[str], extra_env: dict | None = None):
    env = os.environ.copy()
    project_root = str(ROOT)
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)

def _normalize_symbol(symbol: str) -> str:
    symbol = (symbol or "").strip().upper()
    VALID = {"AAPL", "TSLA", "^GSPC", "^IXIC"}
    if symbol not in VALID:
        raise ValueError(f"Поддерживаемые символы: {sorted(VALID)}")
    return symbol

def get_prediction(symbol: str, refresh: bool = False) -> dict:
    symbol = _normalize_symbol(symbol)
    if refresh:
        extra = {"ONLY_SYMBOL": symbol}
        _run([sys.executable, "jobs/ingest_prices.py"])
        _run([sys.executable, "jobs/build_features.py"], extra)

    extra_env = {"ACTION": "infer", "HORIZON_DAYS": "5", "ONLY_SYMBOL": symbol}
    _run([sys.executable, "jobs/train_baseline.py"], extra_env)

    df = pd.read_csv(PREDICTIONS_FILE)
    row = df[df["symbol"].str.upper() == symbol].iloc[-1].to_dict()

    pred = {
        "symbol": symbol,
        "name_ru": {"AAPL": "Apple Inc.", "TSLA": "Tesla Inc.", "^GSPC": "S&P 500", "^IXIC": "Nasdaq Composite"}.get(symbol, symbol),
        "asof_date": row["asof_date"],
        "p_up": round(float(row["p_up"]), 4),
        "vol_pred": round(float(row["vol_pred"]), 4),
    }

    pred = risk_manager.add_to_prediction(pred)

    # === РЕАЛЬНЫЙ SHAP (исправленная обработка nested list) ===
    shap_file = ARTIFACTS_DIR / f"shap_{symbol}_direction.json"
    if shap_file.exists():
        with open(shap_file, encoding="utf-8") as f:
            data = json.load(f)
        
        shap_values = data["shap_values"]
        # Если это список списков — берём первый (или единственный) внутренний список
        if isinstance(shap_values, list) and len(shap_values) > 0 and isinstance(shap_values[0], list):
            shap_values = shap_values[0]

        pred["shap_values"] = shap_values
        pred["shap_feature_names"] = data["feature_names"]
        pred["shap_base_value"] = data["base_value"]

        # Топ-5 факторов
        top = sorted(
            zip(data["feature_names"], shap_values),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        pred["shap_top_factors_ru"] = [f"{name} ({val:+.4f})" for name, val in top]
    else:
        pred["shap_top_factors_ru"] = ["SHAP пока не посчитан"]

    return pred