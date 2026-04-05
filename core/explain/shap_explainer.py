import shap
import numpy as np
import pandas as pd
import json
from pathlib import Path
import warnings
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

def compute_and_save_shap(model, X_latest: np.ndarray, feature_names: list, symbol_task: str):
    """Финальная версия SHAP — поддерживает VotingClassifier, Pipeline, LogReg, XGBoost, LightGBM"""
    print(f"[SHAP DEBUG] Запуск для {symbol_task}...")

    try:
        # Если это Pipeline — берём только саму модель
        if isinstance(model, Pipeline):
            model = model.named_steps[list(model.named_steps.keys())[-1]]

        # === ОБРАБОТКА ГИБРИДНОЙ МОДЕЛИ (VotingClassifier) ===
        if isinstance(model, VotingClassifier):
            print(f"[SHAP] Обнаружен VotingClassifier — усредняем SHAP от всех моделей")
            shap_values_list = []
            base_values = []

            for name, estimator in model.named_estimators_.items():
                if isinstance(estimator, Pipeline):
                    estimator = estimator.named_steps[list(estimator.named_steps.keys())[-1]]

                if isinstance(estimator, LogisticRegression):
                    explainer = shap.LinearExplainer(estimator, X_latest)
                else:
                    explainer = shap.TreeExplainer(estimator)

                sv = explainer.shap_values(X_latest)
                if isinstance(sv, list):
                    sv = sv[1] if len(sv) > 1 else sv[0]

                shap_values_list.append(sv)
                base_values.append(float(explainer.expected_value) if not isinstance(explainer.expected_value, np.ndarray) else float(explainer.expected_value[0]))

            shap_values = np.mean(shap_values_list, axis=0)
            base_value = np.mean(base_values)

        # === Обычные модели ===
        else:
            if hasattr(model, "feature_importances_") or hasattr(model, "estimators_") or "tree" in str(type(model)).lower():
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_latest)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                explainer = shap.LinearExplainer(model, X_latest)
                shap_values = explainer.shap_values(X_latest)

            base_value = float(explainer.expected_value) if not isinstance(explainer.expected_value, np.ndarray) else float(explainer.expected_value[0])

        # Сохраняем результат
        result = {
            "symbol_task": symbol_task,
            "shap_values": shap_values.tolist() if hasattr(shap_values, "tolist") else list(shap_values),
            "feature_names": feature_names,
            "base_value": float(base_value),
            "generated_at": pd.Timestamp.utcnow().isoformat()
        }

        filepath = ARTIFACTS / f"shap_{symbol_task}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[SHAP SUCCESS] Файл сохранён: {filepath}")
        return result

    except Exception as e:
        print(f"[SHAP ERROR] Не удалось сохранить {symbol_task}: {e}")
        return None