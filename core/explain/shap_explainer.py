import shap
import numpy as np
import pandas as pd
import json
from pathlib import Path
import pickle
import warnings

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

def compute_and_save_shap(model, X_latest: np.ndarray, feature_names: list, symbol_task: str):
    """Улучшенный SHAP с поддержкой линейных моделей + отладка"""
    print(f"[SHAP DEBUG] Запуск для {symbol_task}...")

    try:
        # Для моделей на основе деревьев
        if hasattr(model, "estimators_") or "tree" in str(type(model)).lower() or hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_latest)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            # Для линейных моделей (LOGREG, Ridge и т.д.)
            explainer = shap.LinearExplainer(model, X_latest)
            shap_values = explainer.shap_values(X_latest)

        base_value = float(explainer.expected_value) if not isinstance(explainer.expected_value, np.ndarray) else float(explainer.expected_value[0])

        result = {
            "symbol_task": symbol_task,
            "shap_values": shap_values.tolist() if hasattr(shap_values, "tolist") else list(shap_values),
            "feature_names": feature_names,
            "base_value": base_value,
            "generated_at": pd.Timestamp.utcnow().isoformat()
        }

        filepath = ARTIFACTS / f"shap_{symbol_task}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[SHAP SUCCESS] Сохранён файл: {filepath}")
        return result

    except Exception as e:
        print(f"[SHAP ERROR] Не удалось сохранить {symbol_task}: {e}")
        return None