import asyncio
import sys
import json
from datetime import datetime
from pathlib import Path

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram import F
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Подключаем ядро
sys.path.insert(0, "/Users/asadbekikromov/Documents/GitHub/Master_of_Degree")
from services.predict import get_prediction

BOT_TOKEN = "8604575048:AAGPCScLAVACUn2YzNBtN5GMvEkLUm6CLSk"

USERS_FILE = Path("/Users/asadbekikromov/Documents/GitHub/Master_of_Degree/users.json")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
scheduler = AsyncIOScheduler()

# ==================== КЛАВИАТУРА ====================
keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="AAPL"), KeyboardButton(text="TSLA")],
        [KeyboardButton(text="^GSPC"), KeyboardButton(text="^IXIC")],
        [KeyboardButton(text="Все прогнозы")],
    ],
    resize_keyboard=True,
    persistent=True
)

def load_users():
    if USERS_FILE.exists():
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

users = load_users()

def update_user_info(message: types.Message):
    user = message.from_user
    chat_id = str(message.chat.id)
    
    if chat_id not in users:
        users[chat_id] = {
            "username": user.username,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "language_code": user.language_code,
            "subscribed_at": datetime.now().isoformat(),
            "total_predictions": 0
        }
    
    users[chat_id]["last_active"] = datetime.now().isoformat()
    save_users(users)

# ====================== СТАРТ ======================
@dp.message(Command("start"))
async def start_handler(message: types.Message):
    update_user_info(message)
    await message.answer(
        "👋 Добро пожаловать!\n\n"
        "Нажми кнопку ниже, чтобы получить прогноз:",
        reply_markup=keyboard
    )

@dp.message(Command("me"))
async def me_handler(message: types.Message):
    update_user_info(message)
    chat_id = str(message.chat.id)
    u = users.get(chat_id, {})
    text = f"""
📋 <b>Твоя информация</b>

🆔 ID: <code>{chat_id}</code>
👤 Имя: {u.get('first_name', '—')}
📛 Username: @{u.get('username', '—')}
🌍 Язык: {u.get('language_code', '—')}
📅 Подписка: {u.get('subscribed_at', '—')[:10]}
🕒 Последняя активность: {u.get('last_active', '—')[:16]}
📊 Запросов: {u.get('total_predictions', 0)}
    """.strip()
    await message.answer(text, parse_mode="HTML")

# ====================== ПРОГНОЗ ПО КНОПКАМ ======================
@dp.message(F.text.in_({"AAPL", "TSLA", "^GSPC", "^IXIC", "Все прогнозы"}))
async def quick_predict(message: types.Message):
    update_user_info(message)
    chat_id = str(message.chat.id)
    users[chat_id]["total_predictions"] += 1
    save_users(users)

    if message.text == "Все прогнозы":
        symbols = ["AAPL", "TSLA", "^GSPC", "^IXIC"]
        await message.answer("🔄 Отправляю все прогнозы...")
    else:
        symbols = [message.text]

    for symbol in symbols:
        try:
            result = get_prediction(symbol, refresh=True)
            text = f"""
📈 <b>{result['name_ru']} ({result['symbol']})</b>
📅 {result['asof_date']}

🔹 Рекомендация: <b>{result['recommendation_ru']}</b>
🔹 Уверенность: {result['confidence']}
🔹 Риск: {result['risk_label_ru']}

📊 Вероятность роста: <b>{result['p_up']:.1%}</b>
📊 Волатильность: <b>{result['vol_pred']:.2%}</b>

🛡️ {result['risk_summary_ru']}
            """.strip()
            await message.answer(text, parse_mode="HTML")
        except Exception as e:
            await message.answer(f"❌ Ошибка по {symbol}: {str(e)}")

# ====================== ЕЖЕДНЕВНАЯ РАССЫЛКА В 19:00 ======================
async def send_daily_forecast():
    if not users:
        return
    symbols = ["AAPL", "TSLA", "^GSPC", "^IXIC"]
    for symbol in symbols:
        try:
            result = get_prediction(symbol, refresh=False)
            text = f"""
🔔 <b>Ежедневный прогноз • {datetime.now().strftime('%d.%m.%Y')}</b>

📈 {result['name_ru']} ({result['symbol']})
Рекомендация: <b>{result['recommendation_ru']}</b>
Вероятность роста: <b>{result['p_up']:.1%}</b>
Волатильность: <b>{result['vol_pred']:.2%}</b>
            """.strip()
            
            for chat_id in list(users.keys()):
                try:
                    await bot.send_message(int(chat_id), text)
                except:
                    pass
        except:
            pass

# ====================== ЗАПУСК ======================
async def main():
    # Ежедневно в 19:00
    scheduler.add_job(send_daily_forecast, 'cron', hour=19, minute=0)
    
    scheduler.start()

    print("🤖 Бот запущен!")
    print(f"Всего пользователей: {len(users)}")
    print("✅ Ежедневные уведомления настроены на 19:00 каждый день")
    print("Напиши /start в Telegram")

    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())