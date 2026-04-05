from dataclasses import dataclass

@dataclass
class RiskMetrics:
    recommendation_ru: str
    recommendation_en: str
    confidence: str
    risk_label_ru: str
    risk_label_en: str
    position_size: str
    var_5d_approx: float
    no_trade_zone: bool
    expected_return_5d: float
    sharpe_approx: float
    risk_summary_ru: str

class RiskManager:
    def __init__(self):
        # Очень строгая логика
        self.buy_threshold = 0.60      # минимум для "Покупать"
        self.consider_threshold = 0.55 # минимум для "Задуматься о покупке"

    def evaluate(self, p_up: float, vol_pred: float) -> RiskMetrics:
        if p_up >= self.buy_threshold and vol_pred <= 0.022:
            rec_ru = "Покупать"
            rec_en = "Buy"
            confidence = "высокий" if p_up >= 0.65 else "средний"
            risk_label = "низкий"
            position = "8-12% капитала"
            var_5d = round(vol_pred * 2.2, 4)
            no_trade = False
        elif p_up >= self.consider_threshold:
            rec_ru = "Задуматься о покупке"
            rec_en = "Consider Buying"
            confidence = "средний" if p_up >= 0.58 else "низкий"
            risk_label = "средний"
            position = "4-6% капитала"
            var_5d = round(vol_pred * 2.5, 4)
            no_trade = False
        else:
            rec_ru = "Не покупать"
            rec_en = "Do Not Buy"
            confidence = "низкий"
            risk_label = "высокий"
            position = "0% (избегать)"
            var_5d = round(vol_pred * 3.0, 4)
            no_trade = True

        sharpe = round((p_up - 0.5) * 5.0, 2) if vol_pred > 0 else 0.0

        risk_summary_ru = f"Риск: {risk_label} • Позиция: {position} • VaR 5д: {var_5d:.2%}"

        return RiskMetrics(
            recommendation_ru=rec_ru,
            recommendation_en=rec_en,
            confidence=confidence,
            risk_label_ru=risk_label,
            risk_label_en=risk_label,
            position_size=position,
            var_5d_approx=var_5d,
            no_trade_zone=no_trade,
            expected_return_5d=round((p_up - 0.5) * 2.0, 4),
            sharpe_approx=sharpe,
            risk_summary_ru=risk_summary_ru
        )

    def add_to_prediction(self, pred: dict) -> dict:
        metrics = self.evaluate(pred["p_up"], pred["vol_pred"])
        pred.update({
            "recommendation_ru": metrics.recommendation_ru,
            "recommendation_en": metrics.recommendation_en,
            "confidence": metrics.confidence,
            "risk_label_ru": metrics.risk_label_ru,
            "risk_label_en": metrics.risk_label_en,
            "position_size": metrics.position_size,
            "var_5d_approx": metrics.var_5d_approx,
            "no_trade_zone": metrics.no_trade_zone,
            "expected_return_5d": metrics.expected_return_5d,
            "sharpe_approx": metrics.sharpe_approx,
            "risk_summary_ru": metrics.risk_summary_ru,
        })
        return pred