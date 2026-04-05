import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from datetime import datetime

# === ВАЖНЫЕ ИМПОРТЫ ===
from sqlalchemy import create_engine, text

# Подключаем корень проекта
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.predict import get_prediction

st.set_page_config(page_title="Stock Forecast", page_icon="📈", layout="wide")

st.title("📈 Прогноз направления и волатильности акций")
st.markdown("**Гибридная ML-модель** • 5-дневный горизонт • Реальное время")

# Язык
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

if st.sidebar.button("🔄 Обновить данные и прогноз" if is_ru else "🔄 Refresh", use_container_width=True):
    with st.spinner("Обновление данных..."):
        result = get_prediction(symbol, refresh=True)
    st.success("✅ Обновлено!" if is_ru else "✅ Updated!")
else:
    result = get_prediction(symbol, refresh=False)

# ==================== ДАШБОРД ====================
st.subheader(f"{symbols[symbol]} • {result['asof_date']}")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Рекомендация", result["recommendation_ru"] if is_ru else result["recommendation_en"])
with c2:
    st.metric("Уверенность", result["confidence"].capitalize())
with c3:
    st.metric("Риск", result["risk_label_ru"] if is_ru else result["risk_label_en"])

st.info(f"**Вероятность роста:** {result['p_up']:.1%} | **Ожидаемая волатильность (5 дней):** {result['vol_pred']:.2%}")

st.subheader("🛡️ Risk Management")
st.success(result["risk_summary_ru"] if is_ru else result["risk_summary_ru"])

# ==================== ГРАФИК ====================
st.subheader("📈 График цены + прогнозный коридор на 5 дней")

engine = create_engine("postgresql+psycopg://stock:stockpass@localhost:5432/stockdb")
df_price = pd.read_sql_query(
    text("SELECT date, close FROM market_ohlcv WHERE symbol = :sym ORDER BY date DESC LIMIT 60"),
    engine, params={"sym": symbol}
)
df_price = df_price.sort_values("date")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_price["date"], y=df_price["close"], 
                         name="Историческая цена", line=dict(color="#22c55e", width=2)))

last_close = df_price["close"].iloc[-1]
dates_future = pd.date_range(start=df_price["date"].iloc[-1], periods=6, freq="B")[1:]
upper = [last_close * (1 + result["vol_pred"] * (i/2)) for i in range(1, 6)]
lower = [last_close * (1 - result["vol_pred"] * (i/2)) for i in range(1, 6)]

fig.add_trace(go.Scatter(x=dates_future, y=upper, mode="lines", 
                         line=dict(color="rgba(34,197,94,0.4)"), name="Верхний коридор"))
fig.add_trace(go.Scatter(x=dates_future, y=lower, mode="lines", 
                         line=dict(color="rgba(234,179,8,0.4)"), name="Нижний коридор", fill="tonexty"))

fig.update_layout(height=550, template="plotly_dark", 
                  title=f"{symbol} — Последние 60 дней + прогноз на 5 дней")
st.plotly_chart(fig, use_container_width=True)

# ==================== SHAP ====================
st.subheader("🔍 Почему модель решила именно так? (SHAP)")

shap_file = Path("artifacts") / f"shap_{symbol}_direction.json"
if shap_file.exists():
    with open(shap_file, encoding="utf-8") as f:
        shap_data = json.load(f)
    feature_names = shap_data["feature_names"]
    shap_values = shap_data["shap_values"][0] if isinstance(shap_data["shap_values"][0], list) else shap_data["shap_values"]
    base_value = shap_data["base_value"]

    fig_shap = go.Figure(go.Waterfall(
        name="SHAP contribution",
        orientation="v",
        measure=["relative"] * len(feature_names) + ["total"],
        x=feature_names + ["Итоговый прогноз"],
        y=shap_values + [base_value],
        text=[f"{v:+.4f}" for v in shap_values] + [f"{base_value + sum(shap_values):+.4f}"],
        textposition="outside",
        connector=dict(line=dict(color="rgba(63, 63, 63, 0.5)"))
    ))
    fig_shap.update_layout(height=600, template="plotly_dark",
                           title="SHAP Waterfall — вклад факторов")
    st.plotly_chart(fig_shap, use_container_width=True)
else:
    st.info("SHAP-график будет доступен после следующего полного обновления модели.")

st.caption(f"Последнее обновление: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")