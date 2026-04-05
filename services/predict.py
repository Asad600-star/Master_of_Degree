import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.risk.risk_manager import RiskManager   # ← новая строка

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

risk_manager = RiskManager()   # глобальный экземпляр

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _run(cmd: list[str], extra_env: dict | None = None):
    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)

def _normalize_symbol(symbol: str) -> str:
    symbol = (symbol or "").strip().upper()
    if symbol not in VALID_SYMBOLS:
        raise ValueError(f"Поддерживаемые символы: {sorted(VALID_SYMBOLS)}")
    return symbol

def refresh_data(symbol: str | None = None):
    extra = {"ONLY_SYMBOL": _normalize_symbol(symbol)} if symbol else {}
    _run([sys.executable, "jobs/ingest_prices.py"])
    _run([sys.executable, "jobs/build_features.py"], extra)
    return {"status": "ok", "action": "refresh"}

def get_prediction(symbol: str, refresh: bool = False) -> dict:
    symbol = _normalize_symbol(symbol)
    if refresh:
        refresh_data(symbol)

    # Запускаем inference (как было раньше)
    extra_env = {"ACTION": "infer", "HORIZON_DAYS": "5"}
    if symbol:
        extra_env["ONLY_SYMBOL"] = symbol
    _run([sys.executable, "jobs/train_baseline.py"], extra_env)

    df = pd.read_csv(PREDICTIONS_FILE)
    row = df[df["symbol"].str.upper() == symbol].iloc[-1].to_dict()

    # Базовые значения
    pred = {
        "symbol": symbol,
        "name_ru": SYMBOL_NAME_MAP.get(symbol, symbol),
        "name_en": SYMBOL_NAME_MAP.get(symbol, symbol),
        "asof_date": row["asof_date"],
        "horizon_days": 5,
        "p_up": round(float(row["p_up"]), 4),
        "vol_pred": round(float(row["vol_pred"]), 4),
    }

    # === Risk Management + красивые рекомендации ===
    pred = risk_manager.add_to_prediction(pred)

    # === Подготовка под SHAP (пока топ-факторы из модели) ===
    # Позже мы добавим настоящий SHAP explainer
    pred["shap_top_factors_ru"] = [
        "Динамика рынка (mkt_mom_20)",
        "Уровень страха (VIX)",
        "Предыдущая доходность (return_lag_1)",
        "Волатильность актива",
        "Объём рынка"
    ]
    pred["shap_summary_ru"] = "Модель больше всего смотрит на текущий импульс рынка и уровень VIX."

    # === График (пока красивый placeholder, потом сделаем реальный) ===
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.update_layout(
        title=f"{symbol} — Прогноз на 5 дней ({pred['asof_date']})",
        height=650,
        template="plotly_dark"
    )
    pred["plotly_figure"] = fig   # будет использоваться в сайте и боте

    return pred