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
    """Универсальный SHAP для любых моделей (tree + linear)"""
    try:
        if hasattr(model, "estimators_") or "tree" in str(type(model)).lower():
            explainer = shap.TreeExplainer(model)
        else:
            # Для линейных моделей используем LinearExplainer
            explainer = shap.LinearExplainer(model, X_latest)
        
        shap_values = explainer.shap_values(X_latest)

        if isinstance(shap_values, list):  # многоклассовый случай
            shap_values = shap_values[1]  # positive class

        result = {
            "symbol_task": symbol_task,
            "shap_values": shap_values.tolist() if hasattr(shap_values, "tolist") else list(shap_values),
            "feature_names": feature_names,
            "base_value": float(explainer.expected_value) if not isinstance(explainer.expected_value, np.ndarray) else float(explainer.expected_value[0]),
            "generated_at": pd.Timestamp.utcnow().isoformat()
        }

        with open(ARTIFACTS / f"shap_{symbol_task}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # Сохраняем explainer
        with open(ARTIFACTS / f"explainer_{symbol_task}.pkl", "wb") as f:
            pickle.dump(explainer, f)

        return result
    except Exception as e:
        warnings.warn(f"SHAP failed for {symbol_task}: {e}")
        return None