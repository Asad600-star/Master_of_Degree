import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = "postgresql+psycopg://stock:stockpass@localhost:5432/stockdb"
INITIAL_CAPITAL = 100_000
COMMISSION = 0.001
POSITION_SIZE_PCT = 0.07

engine = create_engine(DATABASE_URL)

def run_backtest():
    print("🚀 Финальная рабочая версия backtest\n")

    # Загружаем предсказания
    preds = pd.read_csv("artifacts/predictions_latest.csv")
    preds['asof_date'] = pd.to_datetime(preds['asof_date'])
    print(f"📌 Предсказаний: {len(preds)}")

    # Загружаем цены
    query = """
        SELECT symbol, date, close 
        FROM market_ohlcv 
        WHERE symbol IN ('AAPL', 'TSLA', '^GSPC', '^IXIC')
        ORDER BY symbol, date
    """
    prices = pd.read_sql(query, engine)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"📌 Цен в базе: {len(prices)} строк")

    results = []
    capital = INITIAL_CAPITAL

    for _, row in preds.iterrows():
        symbol = row['symbol']
        pred_date = row['asof_date']
        p_up = row['p_up']

        # Цена входа
        entry_row = prices[(prices['symbol'] == symbol) & (prices['date'] == pred_date)]
        if entry_row.empty:
            print(f"⚠️ Нет цены входа для {symbol} на {pred_date.date()}")
            continue
        entry_price = entry_row['close'].iloc[0]

        # Цена выхода = последняя доступная цена в базе
        last_row = prices[prices['symbol'] == symbol].iloc[-1]
        exit_price = last_row['close']
        exit_date = last_row['date']

        # Расчёт доходности
        ret = (exit_price / entry_price) - 1
        trade_return = ret * POSITION_SIZE_PCT * (1 - COMMISSION)
        capital += capital * trade_return

        results.append({
            'symbol': symbol,
            'pred_date': pred_date.date(),
            'exit_date': exit_date.date(),
            'p_up': round(p_up, 4),
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'return_%': round(ret * 100, 2),
            'capital_after': round(capital, 2)
        })

        print(f"✅ {symbol:6} | {pred_date.date()} → {exit_date.date()} | p_up={p_up:.3f} | return={ret*100:+.2f}%")

    if not results:
        print("\n❌ Нет данных для тестирования.")
        return

    df = pd.DataFrame(results)
    total_return = (capital / INITIAL_CAPITAL) - 1
    days = (df['exit_date'].max() - df['pred_date'].min()).days
    ann_return = ((1 + total_return) ** (365 / days) - 1) * 100 if days > 0 else 0
    win_rate = (df['return_%'] > 0).mean() * 100

    print("\n" + "="*80)
    print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ BACKTEST")
    print("="*80)
    print(f"Стартовый капитал     : ${INITIAL_CAPITAL:,.0f}")
    print(f"Финальный капитал     : ${capital:,.0f}")
    print(f"Общая доходность      : {total_return*100:,.2f}%")
    print(f"Доходность годовая    : {ann_return:,.2f}%")
    print(f"Количество сделок     : {len(df)}")
    print(f"Win rate              : {win_rate:.1f}%")
    print(f"Период тестирования   : {days} дней")
    print("="*80)

    df.to_csv("artifacts/backtest_results.csv", index=False)
    print("✅ Результаты сохранены в artifacts/backtest_results.csv")

if __name__ == "__main__":
    run_backtest()