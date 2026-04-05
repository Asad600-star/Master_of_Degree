import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.graph_objects as go
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.predict import get_prediction
from sqlalchemy import create_engine, text

st.set_page_config(page_title="Stock Forecast", page_icon="📈", layout="wide")

st.title("📈 Прогноз направления и волатильности акций")
st.markdown("**Гибридная ML-модель** • 5-дневный горизонт • Реальное время")

lang = st.sidebar.radio("Язык / Language", ["🇷🇺 Русский", "🇬🇧 English"], horizontal=True)
is_ru = lang.startswith("🇷🇺")

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
    with st.spinner("Обновление..."):
        result = get_prediction(symbol, refresh=True)
    st.success("✅ Обновлено!")
else:
    result = get_prediction(symbol, refresh=False)

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

# ==================== ГРАФИК ЦЕНЫ ====================
st.subheader("📈 График цены + прогнозный коридор")

engine = create_engine("postgresql+psycopg://stock:stockpass@localhost:5432/stockdb")
df_price = pd.read_sql_query(
    text("SELECT date, close FROM market_ohlcv WHERE symbol = :sym ORDER BY date DESC LIMIT 60"),
    engine, params={"sym": symbol}
)
df_price = df_price.sort_values("date")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_price["date"], y=df_price["close"], name="Историческая цена", line=dict(color="#22c55e")))

last_close = df_price["close"].iloc[-1]
dates_future = pd.date_range(start=df_price["date"].iloc[-1], periods=6, freq="B")[1:]
upper = [last_close * (1 + result["vol_pred"] * (i/2)) for i in range(1,6)]
lower = [last_close * (1 - result["vol_pred"] * (i/2)) for i in range(1,6)]

fig.add_trace(go.Scatter(x=dates_future, y=upper, mode="lines", line=dict(color="rgba(34,197,94,0.4)"), name="Верхний коридор"))
fig.add_trace(go.Scatter(x=dates_future, y=lower, mode="lines", line=dict(color="rgba(234,179,8,0.4)"), name="Нижний коридор", fill="tonexty"))

fig.update_layout(height=500, template="plotly_dark", title=f"{symbol} — Последние 60 дней + прогноз")
st.plotly_chart(fig, use_container_width=True)

# ==================== КРАСИВЫЙ SHAP ====================
st.subheader("🔍 Почему модель решила именно так? (SHAP)")

shap_file = Path("artifacts") / f"shap_{symbol}_direction.json"
if shap_file.exists():
    with open(shap_file, encoding="utf-8") as f:
        data = json.load(f)
    
    shap_values = data["shap_values"]
    if isinstance(shap_values[0], list):
        shap_values = shap_values[0]
    
    # Берём только топ-12 самых важных факторов
    top = sorted(zip(data["feature_names"], shap_values), key=lambda x: abs(x[1]), reverse=True)[:12]
    names = [name for name, _ in top]
    values = [val for _, val in top]

    # Горизонтальный барчарт (гораздо читаемее)
    colors = ["#22c55e" if v > 0 else "#ef4444" for v in values]
    fig_shap = go.Figure(go.Bar(
        y=names[::-1],
        x=values[::-1],
        orientation='h',
        marker_color=colors[::-1],
        text=[f"{v:+.4f}" for v in values[::-1]],
        textposition="auto"
    ))
    fig_shap.update_layout(
        height=500,
        template="plotly_dark",
        title="Топ-12 факторов, которые повлияли на решение модели",
        xaxis_title="Вклад в вероятность роста",
        yaxis_title=""
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    # Таблица для дополнительной ясности
    st.write("**Топ факторов (по силе влияния):**")
    table_data = {"Фактор": names, "Вклад": [f"{v:+.4f}" for v in values]}
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)

else:
    st.info("SHAP-график будет доступен после следующего полного обновления модели.")

st.caption(f"Последнее обновление: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")