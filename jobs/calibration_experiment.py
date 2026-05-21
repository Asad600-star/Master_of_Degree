"""Calibration experiment for Article 2: Platt vs Isotonic PAV vs No calibration.

Для каждого инструмента ∈ {AAPL, TSLA, ^GSPC, ^IXIC} и каждого walk-forward фолда:
1. Обучить LightGBM-классификатор на Train.
2. Получить сырые вероятности на Val и Test.
3. Подобрать 3 калибратора по Val:
   • no_calibration   — identity (без калибровки)
   • platt            — сигмоидная (логистическая) регрессия
   • isotonic (PAV)   — изотоническая регрессия (Pool-Adjacent-Violators)
4. Применить к Test и измерить:
   • ECE (Expected Calibration Error) с 10 равными бакетами
   • Brier Score
   • log loss
   • AUC (должен сохраняться при монотонной калибровке)
5. Сохранить агрегированные результаты + сырые предсказания для reliability diagrams.

Выходы:
  artifacts/calibration_metrics.csv  — сводка по складкам
  artifacts/calibration_raw.csv      — сырые (y, p_raw, p_platt, p_iso) для reliability
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from sqlalchemy import create_engine, text

W_TRAIN = 1200
W_VAL = 126
W_TEST = 126
STEP = 63
HORIZON = 5
RANDOM_STATE = 42

SYMBOLS = ["AAPL", "TSLA", "^GSPC", "^IXIC"]

FEATURE_COLS = [
    "open", "high", "low", "close", "volume", "return_1d", "log_return",
    "sma_5", "volatility_5", "sma_10", "volatility_10", "sma_20", "volatility_20",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_4", "return_lag_5",
    "mkt_return_1d", "mkt_log_return", "mkt_mom_5", "mkt_mom_10", "mkt_mom_20",
    "mkt_vol_20", "vix_level", "vix_return_1d", "vix_change_1d",
    "irx_level", "irx_change_1d", "tnx_level", "tnx_change_1d",
]


def load_features(engine, symbol):
    q = text("SELECT * FROM features_daily WHERE symbol = :s ORDER BY date")
    return pd.read_sql_query(q, con=engine, params={"s": symbol}, parse_dates=["date"])


def compute_direction_target(df: pd.DataFrame, k: int) -> pd.Series:
    """d_{t+k} = 1[C_{t+k} >= C_t]."""
    return (df["close"].shift(-k) / df["close"] - 1.0 >= 0).astype(int)


# ─── Калибраторы ───
class NoCalibrator:
    """Идентичная функция (без калибровки)."""
    def fit(self, y, p): return self
    def predict(self, p): return np.clip(p, 1e-7, 1 - 1e-7)


class PlattCalibrator:
    """Сигмоидная (Platt) калибровка."""
    def __init__(self):
        self.lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)

    def fit(self, y, p):
        # Train sigmoid on (logit(p), y)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        x = np.log(p / (1 - p)).reshape(-1, 1)
        self.lr.fit(x, y)
        return self

    def predict(self, p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        x = np.log(p / (1 - p)).reshape(-1, 1)
        return self.lr.predict_proba(x)[:, 1]


class IsotonicCalibrator:
    """Изотоническая PAV-калибровка."""
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, y, p):
        self.iso.fit(p, y)
        return self

    def predict(self, p):
        return np.clip(self.iso.predict(p), 1e-7, 1 - 1e-7)


# ─── Метрики калибровки ───
def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE: weighted average |confidence - accuracy| across bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    indices = np.digitize(p, bins[1:-1])
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = indices == b
        n_b = mask.sum()
        if n_b == 0:
            continue
        conf = p[mask].mean()
        acc = y_true[mask].mean()
        ece += (n_b / n) * abs(conf - acc)
    return float(ece)


def reliability_curve(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins + 1)
    indices = np.digitize(p, bins[1:-1])
    accs, confs, counts = [], [], []
    for b in range(n_bins):
        mask = indices == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        confs.append(float(p[mask].mean()))
        accs.append(float(y_true[mask].mean()))
        counts.append(n_b)
    return confs, accs, counts


def compute_calibration_metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    p_clip = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "ece": expected_calibration_error(y_true, p_clip),
        "brier": float(brier_score_loss(y_true, p_clip)),
        "log_loss": float(log_loss(y_true, p_clip)),
        "auc": float(roc_auc_score(y_true, p_clip)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def main():
    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    rows_metrics = []
    rows_raw = []

    for symbol in SYMBOLS:
        print(f"\n[INFO] Loading {symbol} ...")
        df = load_features(engine, symbol)
        df = df.sort_values("date").reset_index(drop=True)
        df["target_dir"] = compute_direction_target(df, HORIZON)
        cols = FEATURE_COLS + ["target_dir"]
        df = df[cols].dropna().reset_index(drop=True)
        print(f"  {symbol}: {len(df)} usable rows")

        fold = 0
        i = W_TRAIN
        while i + W_VAL + W_TEST <= len(df):
            X_tr = df.iloc[:i][FEATURE_COLS].values
            y_tr = df.iloc[:i]["target_dir"].values
            X_val = df.iloc[i : i + W_VAL][FEATURE_COLS].values
            y_val = df.iloc[i : i + W_VAL]["target_dir"].values
            X_te = df.iloc[i + W_VAL : i + W_VAL + W_TEST][FEATURE_COLS].values
            y_te = df.iloc[i + W_VAL : i + W_VAL + W_TEST]["target_dir"].values

            # Skip degenerate folds (single-class)
            if len(np.unique(y_val)) < 2 or len(np.unique(y_te)) < 2:
                fold += 1; i += STEP; continue

            # Train LightGBM
            try:
                clf = LGBMClassifier(
                    n_estimators=400, learning_rate=0.05, max_depth=6,
                    num_leaves=31, subsample=0.85, colsample_bytree=0.85,
                    reg_lambda=1.0, reg_alpha=0.1,
                    class_weight="balanced", random_state=RANDOM_STATE,
                    n_jobs=-1, verbose=-1,
                )
                clf.fit(X_tr, y_tr)
                p_val = clf.predict_proba(X_val)[:, 1]
                p_te_raw = clf.predict_proba(X_te)[:, 1]
            except Exception as e:
                print(f"  [WARN] {symbol} fold={fold}: {e}")
                fold += 1; i += STEP; continue

            # Fit 3 calibrators on val
            calibrators = {
                "no_cal":   NoCalibrator().fit(y_val, p_val),
                "platt":    PlattCalibrator().fit(y_val, p_val),
                "isotonic": IsotonicCalibrator().fit(y_val, p_val),
            }
            for name, cal in calibrators.items():
                p_te = cal.predict(p_te_raw)
                m = compute_calibration_metrics(y_te, p_te)
                rows_metrics.append({
                    "symbol": symbol, "fold": fold, "method": name,
                    **m, "n_test": len(y_te),
                })
                # Save raw predictions for global reliability diagrams
                for yi, pi in zip(y_te, p_te):
                    rows_raw.append({"symbol": symbol, "fold": fold, "method": name,
                                      "y": int(yi), "p": float(pi)})

            print(f"  {symbol} fold={fold} ✓")
            fold += 1; i += STEP

    df_m = pd.DataFrame(rows_metrics)
    df_r = pd.DataFrame(rows_raw)
    out_m = ROOT / "artifacts" / "calibration_metrics.csv"
    out_r = ROOT / "artifacts" / "calibration_raw.csv"
    df_m.to_csv(out_m, index=False)
    df_r.to_csv(out_r, index=False)
    print(f"\n[OK] Saved {len(df_m)} metric rows to {out_m}")
    print(f"[OK] Saved {len(df_r)} raw prediction rows to {out_r}")


if __name__ == "__main__":
    main()
