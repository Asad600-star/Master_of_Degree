import shap
import numpy as np
import pandas as pd
import json
from pathlib import Path
import pickle

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

def compute_and_save_shap(model, X_latest: np.ndarray, feature_names: list, symbol: str):
    """Вычисляет SHAP для последнего предсказания и сохраняет"""
    explainer = shap.TreeExplainer(model) if hasattr(model, "estimators_") else shap.Explainer(model)
    shap_values = explainer.shap_values(X_latest)

    # Сохраняем в удобном формате
    result = {
        "symbol": symbol,
        "shap_values": shap_values[0].tolist() if isinstance(shap_values, list) else shap_values.tolist(),
        "feature_names": feature_names,
        "base_value": float(explainer.expected_value) if not isinstance(explainer.expected_value, np.ndarray) else float(explainer.expected_value[0]),
        "generated_at": pd.Timestamp.utcnow().isoformat()
    }

    with open(ARTIFACTS / f"shap_{symbol}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Также сохраняем сам explainer (на всякий случай)
    with open(ARTIFACTS / f"explainer_{symbol}.pkl", "wb") as f:
        pickle.dump(explainer, f)

    return result