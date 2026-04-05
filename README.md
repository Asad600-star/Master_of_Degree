# Master_of_Degree

Master_of_Degree/
├── docker-compose.yml
├── .env
├── requirements.txt
├── README.md                  ← обновим сейчас
│
├── artifacts/                 ← все артефакты модели
│   ├── model_registry_k5.csv
│   ├── predictions_latest.csv
│   ├── shap_*_direction.json
│   └── shap_*_volatility.json
│
├── core/
│   ├── explain/
│   │   └── shap_explainer.py
│   └── risk/
│       └── risk_manager.py    ← только что обновили
│
├── jobs/
│   ├── ingest_prices.py
│   ├── build_features.py
│   └── train_baseline.py      ← с улучшенными моделями
│
├── services/
│   └── predict.py             ← главный сервис предсказаний
│
└── apps/
    └── web/
        └── main.py            ← Streamlit сайт



# 📈 Прогноз направления и волатильности акций с использованием гибридных моделей

**Дипломная работа**  
**Тема:** Разработка программы для прогнозирования направления и волатильности цен акций и фондовых индексов с использованием гибридных моделей

### Основные возможности
- Гибридные ML-модели (LogisticRegression + HistGradientBoosting + XGBoost + LightGBM + VotingClassifier)
- Прогноз на 5 дней вперёд (направление + волатильность)
- SHAP-интерпретация (почему модель решила именно так)
- Risk Management с расчётом VaR, позиции и рекомендациями
- Streamlit веб-интерфейс (RU/EN)
- Автоматическое обновление данных из yfinance + PostgreSQL

### Технологический стек
- **Backend**: Python 3.13, scikit-learn, XGBoost, LightGBM, SQLAlchemy
- **Данные**: yfinance + PostgreSQL
- **Frontend**: Streamlit + Plotly
- **Интерпретируемость**: SHAP
- **Контейнеризация**: Docker Compose

### Как запустить проект

```bash
# 1. Запуск базы данных
docker-compose up -d

# 2. Установка зависимостей
pip install -r requirements.txt

# 3. Полное обновление данных и моделей
python -m jobs.ingest_prices
python -m jobs.build_features
env ACTION=infer TASK=direction python -m jobs.train_baseline
env ACTION=infer TASK=volatility python -m jobs.train_baseline

# 4. Запуск сайта
streamlit run apps/web/main.py