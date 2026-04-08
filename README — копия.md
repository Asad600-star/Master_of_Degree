# Master of Degree: Прогнозирование направления и волатильности акций

**Тема дипломной работы:**  
Разработка программы для прогнозирования направления и волатильности цен акций и фондовых индексов с использованием гибридных моделей машинного обучения.

---

## 📋 О проекте

Это **полноценная production-система** машинного обучения, которая:
- Прогнозирует **направление** цены на 5 дней вперёд (рост/падение)
- Прогнозирует **волатильность** на 5 дней вперёд
- Выдаёт **рекомендацию** инвестору: «Покупать», «Не покупать», «Задуматься о покупке»
- Рассчитывает **риск-менеджмент** (VaR, размер позиции, уровень риска)
- Объясняет решение модели с помощью **SHAP**
- Имеет удобный **веб-сайт** и **Telegram-бот**

**Анализируемые инструменты:** AAPL, TSLA, ^GSPC (S&P 500), ^IXIC (NASDAQ)

---

## 🛠 Технологический стек

- **Язык:** Python 3.13
- **База данных:** PostgreSQL + Docker
- **Модели:** Гибридный ensemble (LightGBM + XGBoost + Logistic Regression + VotingClassifier)
- **Объяснимость:** SHAP
- **Веб-интерфейс:** Streamlit
- **Telegram-бот:** aiogram + APScheduler
- **Оркестрация:** Docker Compose

---

## 📁 Структура проекта

Master_of_Degree/
├── .env                          ← Основные настройки
├── docker-compose.yml
├── requirements.txt
├── README.md
├── jobs/                         ← Основные скрипты
│   ├── ingest_prices.py          ← Загрузка цен
│   ├── build_features.py         ← Создание признаков
│   ├── train_baseline.py         ← Обучение и inference
│   ├── daily_update.py           ← Ежедневное обновление (главная команда)
│   └── bot.py                    ← Telegram-бот
├── services/
│   └── predict.py                ← Основная функция предсказания
├── core/                         ← Ядро проекта (SHAP + Risk Manager)
├── apps/web/main.py              ← Веб-сайт на Streamlit
├── artifacts/                    ← Все результаты (модели, метрики, предсказания)
├── users.json                    ← Информация о пользователях
└── subscribers.json              ← Подписчики на уведомления


---

## 🚀 Установка и первый запуск

### 1. Запуск базы данных
```bash
docker-compose up -d

2. Активация виртуального окружения
source .venv/bin/activate

3. Установка зависимостей
pip install -r requirements.txt

4. Первый запуск проекта (выполнить один раз)
# 1. Загрузка исторических данных
python -m jobs.ingest_prices

# 2. Создание признаков
python -m jobs.build_features

# 3. Обучение и обновление предсказаний
env ACTION=infer TASK=direction python -m jobs.train_baseline
env ACTION=infer TASK=volatility python -m jobs.train_baseline

📅 Ежедневная работа с проектом
Самая важная команда (запускай каждый день после 19:00):
python -m jobs.daily_update

🔄 Переобучение модели
Переобучать модель рекомендуется 1 раз в 2–4 недели или при значительном изменении рынка.
# Полное переобучение (walk-forward)
env MODE=walk TASK=direction python -m jobs.train_baseline
env MODE=walk TASK=volatility python -m jobs.train_baseline

После переобучения обязательно обнови предсказания:
env ACTION=infer TASK=direction python -m jobs.train_baseline
env ACTION=infer TASK=volatility python -m jobs.train_baseline

🌐 Запуск веб-сайта (Streamlit)
streamlit run apps/web/main.py

🤖 Запуск Telegram-бота
cd /Users/asadbekikromov/Documents/GitHub/Master_of_Degree
source .venv/bin/activate
python jobs/bot.py

🔧 Полезные команды
Команда,                                                            Описание
python -m jobs.ingest_prices,                                       Загрузить/обновить цены
python -m jobs.build_features,                                      Пересчитать признаки
python -m jobs.daily_update,                                        Главное ежедневное обновление
env ACTION=infer TASK=direction python -m jobs.train_baseline,      Обновить предсказания направления
env ACTION=infer TASK=volatility python -m jobs.train_baseline,     Обновить предсказания волатильности

📊 Как интерпретировать результаты
p_up — вероятность роста за 5 дней
vol_pred — ожидаемая волатильность
recommendation_ru — рекомендация модели
confidence — уровень уверенности
risk_summary_ru — рекомендации по риску

Хороший сигнал на покупку:
p_up > 0.60 + риск: низкий + position_size: 8-12%

⚙️ Настройка параметров
Все основные настройки находятся в файле .env:

TRAIN_START_DATE=2022-01-01 — с какой даты обучать модель
HORIZON_DAYS=5 — горизонт прогнозирования
Другие параметры можно менять по необходимости

Автор: Асадбек Икромов
Год: 2026