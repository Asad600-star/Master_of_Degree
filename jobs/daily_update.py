import subprocess
import sys
from pathlib import Path
from datetime import datetime   # ← ЭТУ СТРОКУ ДОБАВИЛИ

print("🚀 Запуск ежедневного обновления Master_of_Degree...")

commands = [
    ["python", "-m", "jobs.ingest_prices"],           # 1. Новые цены
    ["python", "-m", "jobs.build_features"],          # 2. Пересчёт фич
    ["env", "ACTION=infer", "TASK=direction", "python", "-m", "jobs.train_baseline"],  # 3. Направление
    ["env", "ACTION=infer", "TASK=volatility", "python", "-m", "jobs.train_baseline"], # 4. Волатильность
]

for cmd in commands:
    print(f"▶️ Выполняю: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    if result.returncode != 0:
        print(f"❌ Ошибка на шаге: {' '.join(cmd)}")
        sys.exit(1)

print("✅ Ежедневное обновление завершено успешно!")
print(f"Время: {datetime.now().strftime('%d.%m.%Y %H:%M')}")