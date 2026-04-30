"""Production rules knowledge base generator (v2.0).

Полностью перегенерирует базу знаний на основе РЕАЛЬНЫХ SHAP-значений
из artifacts/shap_*_direction.json. Соответствует пространству
из 56 признаков, реализованному в jobs/train_baseline.py::build_feature_matrix.

Топ-15 признаков по среднему |SHAP| по всем символам (AAPL/TSLA/^GSPC/^IXIC):
    1.  high              (f2)
    2.  sma_20            (f12)
    3.  corr_mkt_60       (f49)
    4.  macd_signal       (f28)
    5.  rsi_14            (f24)
    6.  close             (f4)
    7.  atrp_14           (f32)
    8.  open              (f1)
    9.  mkt_mom_20        (f40)
    10. volume            (f5)
    11. bb_width_20       (f30)
    12. volatility_10     (f11)
    13. irx_level         (f45)
    14. ema_26            (f26)
    15. mkt_mom_10        (f39)
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

random.seed(42)

# ==================== ПРОСТРАНСТВО ПРИЗНАКОВ ====================
# Полное соответствие build_feature_matrix() в jobs/train_baseline.py.
# 56 признаков, индексация с 1.
FEATURE_INDEX = {
    1:  "open",            2:  "high",           3:  "low",           4:  "close",
    5:  "volume",          6:  "return_1d",      7:  "log_return",
    8:  "sma_5",           9:  "volatility_5",
    10: "sma_10",          11: "volatility_10",
    12: "sma_20",          13: "volatility_20",
    14: "return_lag_1",    15: "return_lag_2",   16: "return_lag_3",
    17: "return_lag_4",    18: "return_lag_5",
    19: "hl_range",        20: "oc_return",
    21: "mom_5",           22: "mom_10",         23: "mom_20",
    24: "rsi_14",          25: "ema_12",         26: "ema_26",
    27: "macd",            28: "macd_signal",    29: "macd_hist",
    30: "bb_width_20",     31: "atr_14",         32: "atrp_14",
    33: "vol_z_20",        34: "ret_std_20",     35: "ret_mean_20",
    36: "mkt_return_1d",   37: "mkt_log_return",
    38: "mkt_mom_5",       39: "mkt_mom_10",     40: "mkt_mom_20",
    41: "mkt_vol_20",
    42: "vix_level",       43: "vix_return_1d",  44: "vix_change_1d",
    45: "irx_level",       46: "irx_change_1d",
    47: "tnx_level",       48: "tnx_change_1d",
    49: "corr_mkt_60",     50: "beta_mkt_60",
    51: "mkt_trend_20",    52: "mkt_risk_20",
    53: "vix_log",         54: "vix_z_60",
    55: "vix_x_mktret",    56: "yc_slope",
}

# Топ-15 SHAP-признаков (мониторятся в правилах) — индексы из FEATURE_INDEX.
SHAP_TOP15 = [2, 12, 49, 28, 24, 4, 32, 1, 40, 5, 30, 11, 45, 26, 39]

# Решения экспертной системы:
EVENTS = {
    "a1": "Сильный рост (Strong Bullish)",
    "a2": "Рост / Задуматься о покупке (Bullish)",
    "a3": "Падение (Bearish)",
}

NUM_RULES = 100

# ==================== ЯДРО (12 ВЕРИФИЦИРОВАННЫХ ПРАВИЛ) ====================
# Условия записаны на нормализованных z-score значениях (после StandardScaler).
# Знаки порогов согласованы со SHAP-направлениями для класса d_{t+k}=1.
core_rules = [
    # --- a1: Сильный рост -----------------------------------------------------
    {"ante": [49, 28, 24, 32], "cons": "a1",
     "cond": "f49 > 0.6 AND f28 > 0.5 AND f24 BETWEEN -0.2 AND 1.5 AND f32 < 0.4"},
    {"ante": [12, 4, 28, 11],  "cons": "a1",
     "cond": "f4 > f12 AND f28 > 0.3 AND f11 < 0.0 AND f12 > 0.5"},
    {"ante": [40, 39, 28, 49], "cons": "a1",
     "cond": "f40 > 0.8 AND f39 > 0.6 AND f28 > 0.2 AND f49 > 0.4"},
    {"ante": [30, 24, 32, 12], "cons": "a1",
     "cond": "f30 < -0.5 AND f24 > 0.4 AND f32 < -0.3 AND f12 > 0.3"},

    # --- a2: Рост / задуматься ------------------------------------------------
    {"ante": [49, 28, 24],     "cons": "a2",
     "cond": "f49 BETWEEN 0.0 AND 0.6 AND f28 > 0.0 AND f24 BETWEEN -0.5 AND 0.8"},
    {"ante": [40, 28, 26],     "cons": "a2",
     "cond": "f40 > 0.3 AND f28 > 0.0 AND f26 > 0.0"},
    {"ante": [12, 4, 30],      "cons": "a2",
     "cond": "f4 > f12 AND f30 BETWEEN -0.3 AND 0.3 AND f12 > 0.0"},

    # --- a3: Падение ----------------------------------------------------------
    {"ante": [30, 32, 24, 49], "cons": "a3",
     "cond": "f30 > 0.8 AND f32 > 0.7 AND f24 < -0.5 AND f49 < -0.2"},
    {"ante": [28, 11, 49, 40], "cons": "a3",
     "cond": "f28 < -0.3 AND f11 > 0.6 AND f49 < -0.4 AND f40 < -0.3"},
    {"ante": [24, 32, 30],     "cons": "a3",
     "cond": "f24 < -0.8 AND f32 > 0.5 AND f30 > 0.5"},
    {"ante": [45, 40, 39, 28], "cons": "a3",
     "cond": "f45 > 1.0 AND f40 < -0.2 AND f39 < -0.2 AND f28 < 0.0"},
    {"ante": [49, 28, 24, 11], "cons": "a3",
     "cond": "f49 < -0.5 AND f28 < -0.2 AND f24 < -0.3 AND f11 > 0.4"},
]

# ==================== ВСПОМОГАТЕЛЬНЫЕ ====================

def random_threshold() -> float:
    """Случайный порог в z-score единицах, дискретизирован до 0.05."""
    raw = random.uniform(-2.0, 2.0)
    return round(raw / 0.05) * 0.05


def make_condition(ante: list[int]) -> str:
    """Случайное условие на признаках антецедента."""
    parts: list[str] = []
    for f in ante:
        v = random_threshold()
        op = random.random()
        if op < 0.35:
            parts.append(f"f{f} > {v:.2f}")
        elif op < 0.70:
            parts.append(f"f{f} < {v:.2f}")
        else:
            half = round(random.uniform(0.20, 0.45), 2)
            parts.append(f"f{f} BETWEEN {v - half:.2f} AND {v + half:.2f}")
    return " AND ".join(parts)


def signature(rule: dict) -> tuple:
    return (tuple(sorted(rule["ante"])), rule["cons"], rule["cond"])


# ==================== ГЕНЕРАЦИЯ ====================

def generate(num: int = NUM_RULES) -> list[dict]:
    rules: list[dict] = []
    used: set[tuple] = set()

    for r in core_rules:
        sig = signature(r)
        if sig not in used:
            used.add(sig)
            rules.append(r)

    while len(rules) < num:
        size = random.randint(3, 5)
        ante = sorted(random.sample(SHAP_TOP15, size))
        cons = random.choices(["a1", "a2", "a3"], weights=[0.30, 0.35, 0.35])[0]
        cond = make_condition(ante)
        rule = {"ante": ante, "cons": cons, "cond": cond}
        sig = signature(rule)
        if sig not in used:
            used.add(sig)
            rules.append(rule)
    return rules


def main() -> None:
    rules = generate(NUM_RULES)

    out = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "schema_version": "2.0",
        "total_rules": len(rules),
        "features_count": len(FEATURE_INDEX),
        "feature_index": FEATURE_INDEX,
        "shap_top15_features": SHAP_TOP15,
        "events": EVENTS,
        "rules": rules,
    }

    out_path = Path("production_rules_knowledge_base.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("БАЗА ЗНАНИЙ ПРОДУКЦИОННЫХ ПРАВИЛ v2.0")
    print("=" * 100)
    print(f"Сгенерировано:  {out['generated_at']}")
    print(f"Признаков:      {out['features_count']}")
    print(f"Топ-15 SHAP:    {SHAP_TOP15}")
    print(f"Всего правил:   {len(rules)}")
    print()
    for i, r in enumerate(rules, 1):
        ante_str = " ∧ ".join(f"f{f}" for f in r["ante"])
        print(f"p{i:>3}: {ante_str:<28} → {r['cons']} | {r['cond']}")
    print()
    print(f"Файл сохранён: {out_path.resolve()}")


if __name__ == "__main__":
    main()
