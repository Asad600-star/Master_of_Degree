import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class RiskMetrics:
    recommendation_ru: str
    recommendation_en: str
    confidence: str          # высокий / средний / низкий
    risk_label_ru: str
    risk_label_en: str
    position_size_pct: str   # "0-2% капитала" и т.д.
    var_5d_approx: float     # приблизительный 95% VaR на 5 дней
    no_trade_zone: bool      # True = лучше пропустить сделку
    expected_return: float
    sharpe_approx: float     # грубая оценка


class RiskManager:
    """Полноценный risk-management для гибридной модели"""

    def __init__(self, prob_threshold: float = 0.62, vol_high: float = 0.03, vol_medium: float = 0.015):
        self.prob_threshold = prob_threshold
        self.vol_high = vol_high
        self.vol_medium = vol_medium

    def evaluate(self, p_up: float, vol_pred: float, symbol: str) -> RiskMetrics:
        # 1. Направление + уверенность
        if p_up >= self.prob_threshold:
            rec_ru, rec_en = "Покупать", "Buy"
            confidence = "высокий" if p_up >= 0.70 else "средний"
        elif p_up <= (1 - self.prob_threshold):
            rec_ru, rec_en = "Не покупать", "Don't Buy"
            confidence = "высокий" if p_up <= 0.30 else "средний"
        else:
            rec_ru, rec_en = "Задуматься о покупке", "Consider Buying"
            confidence = "низкий"

        # 2. Risk level и Position sizing
        if vol_pred >= self.vol_high:
            risk_ru = risk_en = "высокий"
            position = "0-2% капитала"
            no_trade = True
        elif vol_pred >= self.vol_medium:
            risk_ru = risk_en = "средний"
            position = "3-5% капитала"
            no_trade = False
        else:
            risk_ru = risk_en = "низкий"
            position = "6-8% капитала"
            no_trade = False

        # 3. Примерный VaR 95% на 5 дней (нормальное приближение)
        var_5d = round(vol_pred * 1.65, 4)

        # 4. Ожидаемая доходность и грубый Sharpe
        expected_ret = round((p_up - 0.5) * 2 * vol_pred, 4)   # очень грубая оценка
        sharpe = round(expected_ret / (vol_pred + 1e-6), 2)

        return RiskMetrics(
            recommendation_ru=rec_ru,
            recommendation_en=rec_en,
            confidence=confidence,
            risk_label_ru=risk_ru,
            risk_label_en=risk_en,
            position_size_pct=position,
            var_5d_approx=var_5d,
            no_trade_zone=no_trade,
            expected_return=expected_ret,
            sharpe_approx=sharpe,
        )

    def add_to_prediction(self, pred_dict: dict) -> dict:
        """Добавляет риск-метрики прямо в словарь предсказания"""
        rm = self.evaluate(
            p_up=pred_dict["p_up"],
            vol_pred=pred_dict["vol_pred"],
            symbol=pred_dict["symbol"]
        )
        pred_dict.update({
            "recommendation_ru": rm.recommendation_ru,
            "recommendation_en": rm.recommendation_en,
            "confidence": rm.confidence,
            "risk_label_ru": rm.risk_label_ru,
            "risk_label_en": rm.risk_label_en,
            "position_size": rm.position_size_pct,
            "var_5d_approx": rm.var_5d_approx,
            "no_trade_zone": rm.no_trade_zone,
            "expected_return_5d": rm.expected_return,
            "sharpe_approx": rm.sharpe_approx,
            "risk_summary_ru": f"Риск: {rm.risk_label_ru} • Позиция: {rm.position_size_pct} • VaR 5д: {rm.var_5d_approx:.2%}",
        })
        return pred_dict