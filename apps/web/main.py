import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from datetime import datetime

# Подключаем корень проекта
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.predict import get_prediction

st.set_page_config(page_title="Stock Forecast", page_icon="📈", layout="wide")

st.title("📈 Прогноз направления и волатильности акций")
st.markdown("**Гибридная ML-модель** • 5-дневный горизонт • Реальное время")

# Выбор языка
lang = st.sidebar.radio("Язык / Language", ["🇷🇺 Русский", "🇬🇧 English"], horizontal=True)
is_ru = lang.startswith("🇷🇺")

# Выбор символа
symbols = {
    "AAPL": "Apple Inc. (AAPL)",
    "TSLA": "Tesla Inc. (TSLA)",
    "^GSPC": "S&P 500 (^GSPC)",
    "^IXIC": "Nasdaq Composite (^IXIC)"
}

symbol = st.sidebar.selectbox(
    "Выберите инструмент" if is_ru else "Select instrument",
    options=list(symbols.keys()),
    format_func=lambda x: symbols[x]
)

col_refresh = st.sidebar.columns([1, 1])
if col_refresh[0].button("🔄 Обновить данные" if is_ru else "🔄 Refresh", use_container_width=True):
    with st.spinner("Обновление данных и прогноза..."):
        result = get_prediction(symbol, refresh=True)
    st.success("✅ Данные и прогноз обновлены!" if is_ru else "✅ Data & forecast updated!")
else:
    result = get_prediction(symbol, refresh=False)

# ==================== ГЛАВНЫЙ ДАШБОРД ====================
st.subheader(f"{symbols[symbol]} • {result['asof_date']}")

c1, c2, c3 = st.columns(3)

with c1:
    rec_color = "#22c55e" if result["recommendation_ru"] == "Покупать" else \
                "#ef4444" if result["recommendation_ru"] == "Не покупать" else "#eab308"
    st.metric("Рекомендация", result["recommendation_ru"] if is_ru else result["recommendation_en"], 
              delta=None, border=True)

with c2:
    st.metric("Уверенность", result["confidence"].capitalize())

with c3:
    st.metric("Риск", result["risk_label_ru"] if is_ru else result["risk_label_en"])

# Основные показатели
st.subheader("📊 Основные показатели")
st.info(f"**Вероятность роста:** {result['p_up']:.1%}  |  **Ожидаемая волатильность (5 дней):** {result['vol_pred']:.2%}")

# Risk Management
st.subheader("🛡️ Risk Management")
st.success(result["risk_summary_ru"] if is_ru else result["risk_summary_ru"])

# ==================== ГРАФИК ЦЕНЫ + ПРОГНОЗ ====================
st.subheader("📈 График цены и прогноз на 5 дней")

# Пока простой красивый график (можно потом усложнить с реальными данными)
fig = make_subplots(rows=1, cols=1)
fig.add_annotation(
    text=f"Прогноз на 5 дней<br>"
         f"Рекомендация: {result['recommendation_ru']}<br>"
         f"p(рост) = {result['p_up']:.1%}<br>"
         f"Волатильность ≈ {result['vol_pred']:.2%}",
    showarrow=False,
    font_size=18,
    bgcolor="#1e2937",
    bordercolor="#64748b",
    borderwidth=2
)
fig.update_layout(height=500, template="plotly_dark", title=f"{symbol} — Последние данные + прогноз")
st.plotly_chart(fig, use_container_width=True)

# ==================== SHAP ====================
st.subheader("🔍 Почему модель решила именно так? (SHAP)")

if result.get("shap_top_factors_ru") and result["shap_top_factors_ru"][0] != "SHAP пока не посчитан":
    for factor in result["shap_top_factors_ru"]:
        st.write(f"• {factor}")
else:
    st.info("SHAP-объяснение будет доступно после следующего полного обновления модели (нажми «Обновить данные»).")

st.caption(f"Последнее обновление: {datetime.now().strftime('%d.%m.%Y %H:%M')}")

# Кнопка полного обновления всех символов
if st.button("🔄 Полное обновление всех 4 инструментов" if is_ru else "🔄 Full refresh all 4 instruments"):
    with st.spinner("Обновляем все символы..."):
        for sym in symbols.keys():
            get_prediction(sym, refresh=True)
    st.success("Все инструменты обновлены!")