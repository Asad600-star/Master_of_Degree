import os
import subprocess
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from core.risk.risk_manager import RiskManager

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
PREDICTIONS_FILE = ARTIFACTS_DIR / "predictions_latest.csv"

risk_manager = RiskManager()

def _run(cmd: list[str], extra_env: dict | None = None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)

def get_prediction(symbol: str, refresh: bool = False) -> dict:
    symbol = symbol.strip().upper()
    if refresh:
        _run([sys.executable, "jobs/ingest_prices.py"])
        _run([sys.executable, "jobs/build_features.py"])

    extra_env = {"ACTION": "infer", "HORIZON_DAYS": "5", "ONLY_SYMBOL": symbol}
    _run([sys.executable, "jobs/train_baseline.py"], extra_env)

    df = pd.read_csv(PREDICTIONS_FILE)
    row = df[df["symbol"] == symbol].iloc[-1].to_dict()

    pred = {
        "symbol": symbol,
        "name_ru": {"AAPL": "Apple Inc.", "TSLA": "Tesla Inc.", "^GSPC": "S&P 500", "^IXIC": "Nasdaq Composite"}.get(symbol, symbol),
        "asof_date": row["asof_date"],
        "p_up": round(float(row["p_up"]), 4),
        "vol_pred": round(float(row["vol_pred"]), 4),
    }

    pred = risk_manager.add_to_prediction(pred)

    # === SHAP (исправленная обработка для гибридных моделей) ===
    shap_file = ARTIFACTS_DIR / f"shap_{symbol}_direction.json"
    if shap_file.exists():
        with open(shap_file, encoding="utf-8") as f:
            data = json.load(f)

        shap_values = data.get("shap_values", [])
        if isinstance(shap_values, list) and len(shap_values) > 0 and isinstance(shap_values[0], list):
            shap_values = shap_values[0]  # распаковываем nested list

        pred["shap_values"] = shap_values
        pred["shap_feature_names"] = data.get("feature_names", [])
        pred["shap_base_value"] = data.get("base_value", 0.0)

        # Топ-8 самых важных факторов
        top = sorted(zip(pred["shap_feature_names"], shap_values), key=lambda x: abs(x[1]), reverse=True)[:8]
        pred["shap_top_factors_ru"] = [f"{name} ({val:+.4f})" for name, val in top]
    else:
        pred["shap_top_factors_ru"] = ["SHAP пока не посчитан"]

    return pred