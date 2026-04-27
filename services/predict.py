"""High-level prediction service.

Использует:
- закэшированные joblib-модели (если есть) — быстрый путь без переобучения,
- subprocess только когда явно нужен refresh (новая загрузка цен / пересчёт фич).
"""

import os
import subprocess
import sys
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from core.risk.risk_manager import RiskManager

ARTIFACTS_DIR = ROOT / "artifacts"
PREDICTIONS_FILE = ARTIFACTS_DIR / "predictions_latest.csv"

risk_manager = RiskManager()

NAME_MAP = {
    "AAPL": "Apple Inc.",
    "TSLA": "Tesla Inc.",
    "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq Composite",
}


def _run(cmd: list[str], extra_env: dict | None = None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _refresh_prices_and_features():
    _run([sys.executable, "-m", "jobs.ingest_prices"])
    _run([sys.executable, "-m", "jobs.build_features"])


def _run_inference(symbol: str | None = None):
    extra_env = {"ACTION": "infer", "HORIZON_DAYS": "5"}
    if symbol:
        extra_env["ONLY_SYMBOL"] = symbol
    _run([sys.executable, "-m", "jobs.train_baseline"], extra_env)


def get_prediction(symbol: str, refresh: bool = False) -> dict:
    """Возвращает прогноз по символу.

    refresh=True  → обновить цены, пересчитать фичи, заново посчитать инференс.
    refresh=False → попытаться вернуть последнее сохранённое в predictions_latest.csv;
                    если нет строки для символа — запустить инференс на лету.
    """
    symbol = symbol.strip().upper()

    if refresh:
        _refresh_prices_and_features()
        _run_inference(symbol)
    elif not PREDICTIONS_FILE.exists():
        _run_inference(symbol)

    df = pd.read_csv(PREDICTIONS_FILE)
    matched = df[df["symbol"] == symbol]
    if matched.empty:
        # Не было прогноза для этого символа — считаем
        _run_inference(symbol)
        df = pd.read_csv(PREDICTIONS_FILE)
        matched = df[df["symbol"] == symbol]
        if matched.empty:
            raise RuntimeError(f"Нет прогноза для {symbol} даже после инференса")

    row = matched.iloc[-1].to_dict()

    pred = {
        "symbol": symbol,
        "name_ru": NAME_MAP.get(symbol, symbol),
        "asof_date": row["asof_date"],
        "p_up": round(float(row["p_up"]), 4),
        "vol_pred": round(float(row["vol_pred"]), 4),
    }
    pred = risk_manager.add_to_prediction(pred)

    # === SHAP ===
    shap_file = ARTIFACTS_DIR / f"shap_{symbol}_direction.json"
    if shap_file.exists():
        try:
            with open(shap_file, encoding="utf-8") as f:
                data = json.load(f)
            shap_values = data.get("shap_values", [])
            if isinstance(shap_values, list) and len(shap_values) > 0 and isinstance(shap_values[0], list):
                shap_values = shap_values[0]
            pred["shap_values"] = shap_values
            pred["shap_feature_names"] = data.get("feature_names", [])
            pred["shap_base_value"] = data.get("base_value", 0.0)
            top = sorted(zip(pred["shap_feature_names"], shap_values), key=lambda x: abs(x[1]), reverse=True)[:8]
            pred["shap_top_factors_ru"] = [f"{name} ({val:+.4f})" for name, val in top]
        except Exception as e:
            pred["shap_top_factors_ru"] = [f"Ошибка SHAP: {e}"]
    else:
        pred["shap_top_factors_ru"] = ["SHAP пока не посчитан"]

    return pred
