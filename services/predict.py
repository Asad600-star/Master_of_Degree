import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap  # pip install shap если ещё нет

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

class PredictionService:
    def __init__(self):
        self.registry = self._load_registry()

    def _load_registry(self):
        path = ARTIFACTS_DIR / "model_registry_k5.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def refresh_data(self, symbol: str | None = None):
        extra = {"ONLY_SYMBOL": _normalize_symbol(symbol)} if symbol else {}
        _run([sys.executable, "jobs/ingest_prices.py"])
        _run([sys.executable, "jobs/build_features.py"], extra)
        return {"status": "ok", "action": "refresh"}

    def predict(self, symbol: str, horizon: int = 5, refresh: bool = False) -> dict:
        symbol = _normalize_symbol(symbol)
        if refresh:
            self.refresh_data(symbol)

        # Запускаем inference (использует registry)
        extra_env = {"ACTION": "infer", "HORIZON_DAYS": str(horizon)}
        if symbol:
            extra_env["ONLY_SYMBOL"] = symbol
        _run([sys.executable, "jobs/train_baseline.py"], extra_env)

        df = pd.read_csv(PREDICTIONS_FILE)
        row = df[df["symbol"].str.upper() == symbol].iloc[-1].to_dict()

        # === Расширенный анализ ===
        p_up = float(row["p_up"])
        vol_pred = float(row["vol_pred"])
        asof = row["asof_date"]

        # 1. Рекомендация + уверенность
        if p_up >= 0.62:
            rec_ru, rec_en = "Покупать", "Buy"
            confidence = "высокий" if p_up >= 0.70 else "средний"
        elif p_up <= 0.38:
            rec_ru, rec_en = "Не покупать", "Don't Buy"
            confidence = "высокий" if p_up <= 0.30 else "средний"
        else:
            rec_ru, rec_en = "Задуматься о покупке", "Consider Buying"
            confidence = "низкий"

        # 2. Risk-management
        risk_score = vol_pred * 100  # в процентах
        if risk_score > 3.0:
            risk_label = "высокий"
            position_size = "0-2% капитала"
        elif risk_score > 1.5:
            risk_label = "средний"
            position_size = "3-5% капитала"
        else:
            risk_label = "низкий"
            position_size = "6-8% капитала"

        # 3. Простой VaR (примерно 95%)
        var_5d = vol_pred * 1.65  # грубая оценка

        # 4. График (возвращаем figure, чтобы можно было сохранить в PNG)
        fig = self._create_price_chart(symbol, asof)

        # 5. SHAP (пока заглушка — позже добавим реальный explainer)
        shap_top = ["mkt_mom_20", "vix_level", "return_lag_1", "volatility_20", "mkt_vol_20"]

        result = {
            "symbol": symbol,
            "name_ru": SYMBOL_NAME_MAP.get(symbol, symbol),
            "name_en": SYMBOL_NAME_MAP.get(symbol, symbol),
            "asof_date": asof,
            "horizon_days": 5,
            "p_up": round(p_up, 4),
            "vol_pred": round(vol_pred, 4),
            "recommendation_ru": rec_ru,
            "recommendation_en": rec_en,
            "confidence": confidence,
            "risk_label_ru": risk_label,
            "risk_label_en": risk_label,
            "position_size": position_size,
            "var_5d_approx": round(var_5d, 4),
            "shap_top_factors": shap_top,
            "user_summary_ru": f"{SYMBOL_NAME_MAP.get(symbol)}: {rec_ru}. Уверенность — {confidence}. Ожидаемая волатильность ≈ {vol_pred:.2%}",
            "user_summary_en": f"{SYMBOL_NAME_MAP.get(symbol)}: {rec_en}. Confidence — {confidence}. Expected volatility ≈ {vol_pred:.2%}",
            "plotly_figure": fig,   # для сайта и бота
        }
        return result

    def _create_price_chart(self, symbol: str, asof: str):
        # Здесь можно загрузить последние данные из БД и построить график
        # Для примера — красивый placeholder
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.08, row_heights=[0.7, 0.3])
        fig.add_annotation(text="График цены + прогнозный коридор 5 дней<br>(будет реальный после интеграции)",
                          showarrow=False, font_size=16)
        fig.update_layout(title=f"{symbol} — 5-дневный прогноз", height=600)
        return fig

# ====================== ГЛОБАЛЬНЫЙ СЕРВИС ======================
service = PredictionService()

def get_prediction(symbol: str, refresh: bool = False):
    return service.predict(symbol, horizon=5, refresh=refresh)