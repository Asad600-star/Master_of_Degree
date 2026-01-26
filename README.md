# Master_of_Degree

stock-forecast-hybrid/
  apps/
    api/                 # FastAPI
    web/                 # Streamlit
    bot/                 # Telegram bot
  core/
    config/              # env, settings
    db/                  # models, session, migrations
    data/                # ingestion sources, loaders
    features/            # feature engineering
    models/              # ML models, training, evaluation
    risk/                # VaR/CVaR, volatility, position sizing, no-trade
    backtest/            # walk-forward, metrics
    explain/             # SHAP/permutation, explanations
    utils/               # time utils, logging, validation
  jobs/
    daily_update.py      # daily ingestion + train + predict + notify
  tests/
  docker-compose.yml
  .env.example
  README.md