import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime

# ==================== НАСТРОЙКИ ====================
DATABASE_URL = "postgresql+psycopg://stock:stockpass@localhost:5432/stockdb"
INITIAL_CAPITAL = 100_000
COMMISSION = 0.001          # 0.1% комиссия
POSITION_SIZE_PCT = 0.07    # 7% от капитала на сделку (как в risk_manager)

engine = create_engine(DATABASE_URL)

def determine_recommendation(p_up: float) -> str:
    """Рассчитываем рекомендацию по тем же правилам, что и в risk_manager"""
    if p_up >= 0.55:
        return "Покупать"
    elif p_up >= 0.45:
        return "Задуматься о покупке"
    else:
        return "Не покупать"

def run_backtest():
    print("🚀 Запуск backtest модели...\n")

    # 1. Загружаем предсказания
    preds = pd.read_csv("artifacts/predictions_latest.csv")
    preds['asof_date'] = pd.to_datetime(preds['asof_date'])

    # 2. Загружаем реальные цены
    query = """
        SELECT symbol, date, close 
        FROM market_ohlcv 
        WHERE symbol IN ('AAPL', 'TSLA', '^GSPC', '^IXIC')
        ORDER BY date
    """
    prices = pd.read_sql(query, engine)
    prices['date'] = pd.to_datetime(prices['date'])

    # 3. Объединяем
    df = preds.merge(prices, left_on=['symbol', 'asof_date'], right_on=['symbol', 'date'], how='left')
    df = df.sort_values(['symbol', 'asof_date']).reset_index(drop=True)

    # 4. Симуляция торговли
    capital = INITIAL_CAPITAL
    position = 0.0
    entry_price = 0.0
    entry_date = None
    results = []

    for i, row in df.iterrows():
        symbol = row['symbol']
        p_up = row['p_up']
        close = row['close']
        rec = determine_recommendation(p_up)

        # Закрываем позицию через 5 дней
        if position > 0 and (row['asof_date'] - entry_date).days >= 5:
            ret = (close / entry_price) - 1
            capital = capital * (1 + ret * position) * (1 - COMMISSION)
            results.append({
                'date': row['asof_date'],
                'symbol': symbol,
                'action': 'exit',
                'return': ret * 100,
                'capital': capital
            })
            position = 0.0

        # Открываем новую позицию
        if position == 0 and rec == "Покупать":
            position = POSITION_SIZE_PCT
            entry_price = close
            entry_date = row['asof_date']
            results.append({
                'date': row['asof_date'],
                'symbol': symbol,
                'action': 'buy',
                'p_up': p_up,
                'capital': capital
            })

    # Финальный отчёт
    df_res = pd.DataFrame(results)
    total_return = (capital / INITIAL_CAPITAL) - 1
    days = (df_res['date'].max() - df_res['date'].min()).days if not df_res.empty else 0
    ann_return = ((1 + total_return) ** (365 / days) - 1) * 100 if days > 0 else 0

    print("=" * 70)
    print("📊 РЕЗУЛЬТАТЫ BACKTEST")
    print("=" * 70)
    print(f"Стартовый капитал      : ${INITIAL_CAPITAL:,.0f}")
    print(f"Финальный капитал      : ${capital:,.0f}")
    print(f"Общая доходность       : {total_return*100:,.2f}%")
    print(f"Доходность годовая     : {ann_return:,.2f}%")
    print(f"Количество сделок      : {len(df_res[df_res['action']=='buy'])}")
    print(f"Период тестирования    : {days} дней")
    print("=" * 70)

    # Сохраняем результаты
    df_res.to_csv("artifacts/backtest_results.csv", index=False)
    print("✅ Результаты сохранены в artifacts/backtest_results.csv")

if __name__ == "__main__":
    run_backtest()