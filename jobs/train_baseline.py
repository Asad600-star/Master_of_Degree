import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

load_dotenv()

# ==========================================================
# Baselines for financial time series (no leakage)
# Supports tasks via env:
#   TASK=return|direction|volatility
# Horizon via env:
#   HORIZON_DAYS=1..N
# Split mode:
#   MODE=single|walk
#     - single: one fixed split using TRAIN_END/VAL_END
#     - walk  : walk-forward evaluation (recommended)
# ==========================================================

TASK = (os.environ.get("TASK", "return") or "return").strip().lower()
HORIZON_DAYS = int(os.environ.get("HORIZON_DAYS", "5"))
if HORIZON_DAYS < 1:
    raise ValueError("HORIZON_DAYS must be >= 1")

MODE = (os.environ.get("MODE", "walk") or "walk").strip().lower()
if MODE not in ("single", "walk"):
    raise ValueError("MODE must be 'single' or 'walk'")

# Fixed split boundaries (used in MODE=single)
TRAIN_END = (os.environ.get("TRAIN_END", "2022-12-31") or "2022-12-31").strip()
VAL_END = (os.environ.get("VAL_END", "2024-12-31") or "2024-12-31").strip()

# Output
OUT_PATH = Path(
    os.environ.get(
        "BASELINE_OUT",
        f"artifacts/metrics_{MODE}_{TASK}_k{HORIZON_DAYS}.csv",
    )
)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

RETURN_TARGET_MODE = (os.environ.get("RETURN_TARGET_MODE", "log") or "log").strip().lower()
if RETURN_TARGET_MODE not in ("log", "simple"):
    raise ValueError("RETURN_TARGET_MODE must be 'log' or 'simple'")

RANDOM_STATE = int(os.environ.get("SEED", "42"))

# Walk-forward knobs (used in MODE=walk)
WF_MIN_TRAIN_ROWS = int(os.environ.get("WF_MIN_TRAIN_ROWS", "1200"))
WF_VAL_DAYS = int(os.environ.get("WF_VAL_DAYS", "126"))   # ~6 months trading days
WF_TEST_DAYS = int(os.environ.get("WF_TEST_DAYS", "126")) # ~6 months trading days
WF_STEP_DAYS = int(os.environ.get("WF_STEP_DAYS", "63"))  # ~3 months step


def get_env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name, default)
    if v is None or str(v).strip() == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return str(v).strip()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return atr


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """No-leakage features computed only from past & present."""
    df = df.copy()

    # Basic price/volume derived
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0.0, np.nan)
    df["oc_return"] = (df["close"] - df["open"]) / df["open"].replace(0.0, np.nan)

    # Momentum
    df["mom_5"] = df["close"].pct_change(5)
    df["mom_10"] = df["close"].pct_change(10)
    df["mom_20"] = df["close"].pct_change(20)

    # RSI
    df["rsi_14"] = _rsi(df["close"], 14)

    # EMAs + MACD
    df["ema_12"] = _ema(df["close"], 12)
    df["ema_26"] = _ema(df["close"], 26)
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = _ema(df["macd"], 9)
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger width (20)
    mid = df["close"].rolling(20, min_periods=20).mean()
    std = df["close"].rolling(20, min_periods=20).std(ddof=0)
    df["bb_width_20"] = (2.0 * std) / mid.replace(0.0, np.nan)

    # ATR (volatility proxy)
    df["atr_14"] = _atr(df, 14)
    df["atrp_14"] = df["atr_14"] / df["close"].replace(0.0, np.nan)

    # Volume z-score (20)
    v = df["volume"].astype(float)
    v_mean = v.rolling(20, min_periods=20).mean()
    v_std = v.rolling(20, min_periods=20).std(ddof=0)
    df["vol_z_20"] = (v - v_mean) / v_std.replace(0.0, np.nan)

    # Return distribution (20)
    r = df["return_1d"]
    df["ret_std_20"] = r.rolling(20, min_periods=20).std(ddof=0)
    df["ret_mean_20"] = r.rolling(20, min_periods=20).mean()

    return df


def compute_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Targets without leakage."""
    df = df.copy()

    df["target_return_kd"] = df["close"].shift(-horizon) / df["close"] - 1.0
    df["target_logret_kd"] = np.log(df["close"].shift(-horizon) / df["close"]).replace(
        [np.inf, -np.inf], np.nan
    )
    df["target_direction"] = (df["target_return_kd"] >= 0).astype(int)

    # Realized vol over NEXT k days: std(log_return[t+1..t+k])
    fut = [df["log_return"].shift(-i) for i in range(1, horizon + 1)]
    tmp = pd.concat(fut, axis=1)
    df["target_vol_kd"] = tmp.std(axis=1, ddof=0)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=[
            "target_return_kd",
            "target_logret_kd",
            "target_direction",
            "target_vol_kd",
        ]
    ).reset_index(drop=True)
    return df


def print_reg(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    m = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }
    print(f"[{name}] MAE={m['mae']:.6f} RMSE={m['rmse']:.6f} R2={m['r2']:.4f}")
    return m


def print_clf(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    auc = safe_auc(y_true, y_prob)
    posrate = float(np.mean(y_true == 1))
    print(
        f"[{name}] Acc={acc:.4f} BalAcc={bacc:.4f} F1={f1:.4f} "
        f"AUC={(auc if np.isfinite(auc) else np.nan):.4f} PosRate={posrate:.4f}"
    )
    return {"acc": acc, "balacc": bacc, "f1": f1, "auc": float(auc), "posrate": posrate}


def _best_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 19):
        y_pred = (y_prob >= t).astype(int)
        f1 = float(f1_score(y_true, y_pred))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


@dataclass
class RowReg:
    symbol: str
    mode: str
    fold: int
    split: str
    task: str
    horizon_days: int
    model: str
    n_rows: int
    mae: float
    rmse: float
    r2: float
    extra: str


@dataclass
class RowClf:
    symbol: str
    mode: str
    fold: int
    split: str
    task: str
    horizon_days: int
    model: str
    n_rows: int
    acc: float
    balacc: float
    f1: float
    auc: float
    posrate: float
    threshold: float
    extra: str


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return_1d",
        "log_return",
        "sma_5",
        "volatility_5",
        "sma_10",
        "volatility_10",
        "sma_20",
        "volatility_20",
        "return_lag_1",
        "return_lag_2",
        "return_lag_3",
        "return_lag_4",
        "return_lag_5",
        # engineered
        "hl_range",
        "oc_return",
        "mom_5",
        "mom_10",
        "mom_20",
        "rsi_14",
        "ema_12",
        "ema_26",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_width_20",
        "atr_14",
        "atrp_14",
        "vol_z_20",
        "ret_std_20",
        "ret_mean_20",
    ]
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df, feature_cols


def make_models(task: str):
    if task in ("return", "volatility"):
        ridge = Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=5.0, random_state=RANDOM_STATE))]
        )
        enet = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(alpha=1e-3, l1_ratio=0.2, random_state=RANDOM_STATE, max_iter=20000),
                ),
            ]
        )
        hgb = HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.03,
            max_iter=2000,
            l2_regularization=1.0,
            min_samples_leaf=30,
            random_state=RANDOM_STATE,
        )
        return [
            ("RIDGE", ridge, "alpha=5.0"),
            ("ELASTICNET", enet, "alpha=1e-3 l1=0.2"),
            ("HGB", hgb, "lr=0.03 depth=3 l2=1.0 leaf=30"),
        ]

    if task == "direction":
        logreg = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=5000, C=0.5, class_weight="balanced", random_state=RANDOM_STATE),
                ),
            ]
        )
        hgb = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.03,
            max_iter=2000,
            l2_regularization=1.0,
            min_samples_leaf=30,
            random_state=RANDOM_STATE,
        )
        return [
            ("LOGREG", logreg, "C=0.5 balanced"),
            ("HGB", hgb, "lr=0.03 depth=3 l2=1.0 leaf=30"),
        ]

    raise ValueError(f"Unknown task={task}")


def eval_one_split_reg(sym: str, fold: int, split_name: str, task: str, y_true, y_pred, n_rows: int, model_name: str, extra: str, out: list[RowReg]):
    m = print_reg(f"{model_name}({split_name})", y_true, y_pred)
    out.append(RowReg(sym, MODE, fold, split_name, task, HORIZON_DAYS, model_name, n_rows, **m, extra=extra))


def eval_one_split_clf(sym: str, fold: int, split_name: str, task: str, y_true, y_pred, y_prob, n_rows: int, model_name: str, threshold: float, extra: str, out: list[RowClf]):
    m = print_clf(f"{model_name}({split_name})", y_true, y_pred, y_prob)
    out.append(RowClf(sym, MODE, fold, split_name, task, HORIZON_DAYS, model_name, n_rows, **m, threshold=threshold, extra=extra))


def run_single(df: pd.DataFrame, sym: str) -> tuple[list[RowReg], list[RowClf]]:
    results_reg: list[RowReg] = []
    results_clf: list[RowClf] = []

    # time split
    train = df[df["date"] <= TRAIN_END]
    val = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)]
    test = df[df["date"] > VAL_END]

    if len(train) < 500 or len(val) < 150 or len(test) < 150:
        print(f"[WARN] {sym}: not enough rows after split: train={len(train)} val={len(val)} test={len(test)}")
        return results_reg, results_clf

    df2, feature_cols = build_feature_matrix(df)
    # re-split after NA drop (keeps chronology)
    train = df2[df2["date"] <= TRAIN_END]
    val = df2[(df2["date"] > TRAIN_END) & (df2["date"] <= VAL_END)]
    test = df2[df2["date"] > VAL_END]

    if len(train) < 500 or len(val) < 150 or len(test) < 150:
        print(f"[WARN] {sym}: not enough rows after feature drop: train={len(train)} val={len(val)} test={len(test)}")
        return results_reg, results_clf

    X_train = train[feature_cols].values
    X_val = val[feature_cols].values
    X_test = test[feature_cols].values

    print(f"\n=== {sym} ===")
    print(f"[ROWS] train={len(train)} val={len(val)} test={len(test)}")

    if TASK in ("return", "volatility"):
        ycol = "target_vol_kd" if TASK == "volatility" else ("target_logret_kd" if RETURN_TARGET_MODE == "log" else "target_return_kd")
        y_train = train[ycol].values
        y_val = val[ycol].values
        y_test = test[ycol].values

        # Baselines
        mean_train = float(np.mean(y_train))
        eval_one_split_reg(sym, 0, "val", TASK, y_val, np.zeros_like(y_val), len(val), "BASELINE_ZERO", "", results_reg)
        eval_one_split_reg(sym, 0, "test", TASK, y_test, np.zeros_like(y_test), len(test), "BASELINE_ZERO", "", results_reg)
        eval_one_split_reg(sym, 0, "val", TASK, y_val, np.full_like(y_val, mean_train, dtype=float), len(val), "BASELINE_MEAN_TRAIN", f"mean={mean_train:.6g}", results_reg)
        eval_one_split_reg(sym, 0, "test", TASK, y_test, np.full_like(y_test, mean_train, dtype=float), len(test), "BASELINE_MEAN_TRAIN", f"mean={mean_train:.6g}", results_reg)

        for name, model, extra in make_models(TASK):
            model.fit(X_train, y_train)
            eval_one_split_reg(sym, 0, "val", TASK, y_val, model.predict(X_val), len(val), name, extra, results_reg)
            eval_one_split_reg(sym, 0, "test", TASK, y_test, model.predict(X_test), len(test), name, extra, results_reg)

    else:  # direction
        y_train = train["target_direction"].values
        y_val = val["target_direction"].values
        y_test = test["target_direction"].values

        # Baselines
        eval_one_split_clf(sym, 0, "val", TASK, y_val, np.zeros_like(y_val), np.zeros_like(y_val, dtype=float), len(val), "BASELINE_ALL_ZERO", 0.5, "", results_clf)
        eval_one_split_clf(sym, 0, "test", TASK, y_test, np.zeros_like(y_test), np.zeros_like(y_test, dtype=float), len(test), "BASELINE_ALL_ZERO", 0.5, "", results_clf)

        p = float(np.mean(y_train == 1))
        prob_val = np.full_like(y_val, p, dtype=float)
        prob_test = np.full_like(y_test, p, dtype=float)
        pred_val = (prob_val >= 0.5).astype(int)
        pred_test = (prob_test >= 0.5).astype(int)
        eval_one_split_clf(sym, 0, "val", TASK, y_val, pred_val, prob_val, len(val), "BASELINE_CONST_FROM_TRAIN", 0.5, f"p_train={p:.3f}", results_clf)
        eval_one_split_clf(sym, 0, "test", TASK, y_test, pred_test, prob_test, len(test), "BASELINE_CONST_FROM_TRAIN", 0.5, f"p_train={p:.3f}", results_clf)

        for name, model, extra in make_models(TASK):
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                pv = model.predict_proba(X_val)[:, 1]
                pt = model.predict_proba(X_test)[:, 1]
            else:
                sv = model.decision_function(X_val)
                st = model.decision_function(X_test)
                pv = 1.0 / (1.0 + np.exp(-sv))
                pt = 1.0 / (1.0 + np.exp(-st))

            t_best, f1_best = _best_threshold_f1(y_val, pv)
            eval_one_split_clf(sym, 0, "val", TASK, y_val, (pv >= t_best).astype(int), pv, len(val), name, t_best, f"t_f1={f1_best:.4f} {extra}", results_clf)
            eval_one_split_clf(sym, 0, "test", TASK, y_test, (pt >= t_best).astype(int), pt, len(test), name, t_best, f"t_f1={f1_best:.4f} {extra}", results_clf)

    return results_reg, results_clf


def run_walk(df: pd.DataFrame, sym: str) -> tuple[list[RowReg], list[RowClf]]:
    results_reg: list[RowReg] = []
    results_clf: list[RowClf] = []

    df2, feature_cols = build_feature_matrix(df)
    if df2.empty:
        print(f"[WARN] {sym}: empty after feature drop")
        return results_reg, results_clf

    dates = df2["date"].reset_index(drop=True)
    n = len(df2)

    # Start so we have enough train + val + test
    start_idx = WF_MIN_TRAIN_ROWS
    min_total = WF_MIN_TRAIN_ROWS + WF_VAL_DAYS + WF_TEST_DAYS
    if n < min_total:
        print(f"[WARN] {sym}: not enough rows for walk-forward: n={n}, need>={min_total}")
        return results_reg, results_clf

    # last index where we can still fit val+test
    last_start = n - (WF_VAL_DAYS + WF_TEST_DAYS)

    fold = 0
    i = start_idx
    while i <= last_start:
        # expanding train up to i (exclusive)
        tr = df2.iloc[:i].copy()
        va = df2.iloc[i : i + WF_VAL_DAYS].copy()
        te = df2.iloc[i + WF_VAL_DAYS : i + WF_VAL_DAYS + WF_TEST_DAYS].copy()

        # compute targets *after* split boundaries already fixed by indexing
        tr = compute_targets(tr, HORIZON_DAYS)
        va = compute_targets(va, HORIZON_DAYS)
        te = compute_targets(te, HORIZON_DAYS)

        # After target creation, re-align: we must drop tail rows that lost targets.
        # Ensure we still have reasonable sizes.
        if len(tr) < 500 or len(va) < 80 or len(te) < 80:
            i += WF_STEP_DAYS
            continue

        X_train = tr[feature_cols].values
        X_val = va[feature_cols].values
        X_test = te[feature_cols].values

        print(f"\n=== {sym} | fold={fold} | train_end={dates.iloc[i-1].date()} ===")
        print(f"[ROWS] train={len(tr)} val={len(va)} test={len(te)}")

        if TASK in ("return", "volatility"):
            ycol = "target_vol_kd" if TASK == "volatility" else ("target_logret_kd" if RETURN_TARGET_MODE == "log" else "target_return_kd")
            y_train = tr[ycol].values
            y_val = va[ycol].values
            y_test = te[ycol].values

            mean_train = float(np.mean(y_train))
            eval_one_split_reg(sym, fold, "val", TASK, y_val, np.zeros_like(y_val), len(va), "BASELINE_ZERO", "", results_reg)
            eval_one_split_reg(sym, fold, "test", TASK, y_test, np.zeros_like(y_test), len(te), "BASELINE_ZERO", "", results_reg)
            eval_one_split_reg(sym, fold, "val", TASK, y_val, np.full_like(y_val, mean_train, dtype=float), len(va), "BASELINE_MEAN_TRAIN", f"mean={mean_train:.6g}", results_reg)
            eval_one_split_reg(sym, fold, "test", TASK, y_test, np.full_like(y_test, mean_train, dtype=float), len(te), "BASELINE_MEAN_TRAIN", f"mean={mean_train:.6g}", results_reg)

            for name, model, extra in make_models(TASK):
                model.fit(X_train, y_train)
                eval_one_split_reg(sym, fold, "val", TASK, y_val, model.predict(X_val), len(va), name, extra, results_reg)
                eval_one_split_reg(sym, fold, "test", TASK, y_test, model.predict(X_test), len(te), name, extra, results_reg)

        else:
            y_train = tr["target_direction"].values
            y_val = va["target_direction"].values
            y_test = te["target_direction"].values

            eval_one_split_clf(sym, fold, "val", TASK, y_val, np.zeros_like(y_val), np.zeros_like(y_val, dtype=float), len(va), "BASELINE_ALL_ZERO", 0.5, "", results_clf)
            eval_one_split_clf(sym, fold, "test", TASK, y_test, np.zeros_like(y_test), np.zeros_like(y_test, dtype=float), len(te), "BASELINE_ALL_ZERO", 0.5, "", results_clf)

            p = float(np.mean(y_train == 1))
            pv0 = np.full_like(y_val, p, dtype=float)
            pt0 = np.full_like(y_test, p, dtype=float)
            eval_one_split_clf(sym, fold, "val", TASK, y_val, (pv0 >= 0.5).astype(int), pv0, len(va), "BASELINE_CONST_FROM_TRAIN", 0.5, f"p_train={p:.3f}", results_clf)
            eval_one_split_clf(sym, fold, "test", TASK, y_test, (pt0 >= 0.5).astype(int), pt0, len(te), "BASELINE_CONST_FROM_TRAIN", 0.5, f"p_train={p:.3f}", results_clf)

            for name, model, extra in make_models(TASK):
                model.fit(X_train, y_train)
                if hasattr(model, "predict_proba"):
                    pv = model.predict_proba(X_val)[:, 1]
                    pt = model.predict_proba(X_test)[:, 1]
                else:
                    sv = model.decision_function(X_val)
                    st = model.decision_function(X_test)
                    pv = 1.0 / (1.0 + np.exp(-sv))
                    pt = 1.0 / (1.0 + np.exp(-st))

                t_best, f1_best = _best_threshold_f1(y_val, pv)
                eval_one_split_clf(sym, fold, "val", TASK, y_val, (pv >= t_best).astype(int), pv, len(va), name, t_best, f"t_f1={f1_best:.4f} {extra}", results_clf)
                eval_one_split_clf(sym, fold, "test", TASK, y_test, (pt >= t_best).astype(int), pt, len(te), name, t_best, f"t_f1={f1_best:.4f} {extra}", results_clf)

        fold += 1
        i += WF_STEP_DAYS

    return results_reg, results_clf


def main() -> None:
    db_url = get_env("DATABASE_URL")

    only_symbol = os.environ.get("ONLY_SYMBOL")
    only_symbol = only_symbol.strip() if only_symbol else None

    engine = create_engine(db_url, pool_pre_ping=True)

    if only_symbol:
        symbols = [only_symbol]
    else:
        symbols = pd.read_sql_query(
            "SELECT DISTINCT symbol FROM features_daily ORDER BY symbol",
            con=engine,
        )["symbol"].tolist()

    if not symbols:
        raise RuntimeError("No symbols in features_daily. Run build_features.py first.")

    print(f"[INFO] MODE={MODE} TASK={TASK} HORIZON_DAYS={HORIZON_DAYS} RETURN_TARGET_MODE={RETURN_TARGET_MODE}")
    print(f"[INFO] Symbols={symbols}")
    if MODE == "single":
        print(f"[INFO] Split: train_end={TRAIN_END}, val_end={VAL_END}")
    else:
        print(
            f"[INFO] WalkForward: min_train_rows={WF_MIN_TRAIN_ROWS} val_days={WF_VAL_DAYS} test_days={WF_TEST_DAYS} step_days={WF_STEP_DAYS}"
        )

    all_reg: list[RowReg] = []
    all_clf: list[RowClf] = []

    for sym in symbols:
        df = pd.read_sql_query(
            text(
                """
                SELECT symbol, date,
                       open, high, low, close, volume,
                       return_1d, log_return,
                       sma_5, volatility_5,
                       sma_10, volatility_10,
                       sma_20, volatility_20,
                       return_lag_1, return_lag_2, return_lag_3, return_lag_4, return_lag_5
                FROM features_daily
                WHERE symbol = :symbol
                ORDER BY date
                """
            ),
            con=engine,
            params={"symbol": sym},
            parse_dates=["date"],
        )

        if df.empty:
            print(f"[WARN] {sym}: empty, skip")
            continue

        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        df = add_technical_features(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = compute_targets(df, HORIZON_DAYS)

        if MODE == "single":
            r, c = run_single(df, sym)
        else:
            r, c = run_walk(df, sym)

        all_reg.extend(r)
        all_clf.extend(c)

    if TASK in ("return", "volatility"):
        if not all_reg:
            raise RuntimeError("No results produced. Check data size/splits.")
        out_df = pd.DataFrame([asdict(r) for r in all_reg])
        out_df.to_csv(OUT_PATH, index=False)

        # quick summary across folds (test only)
        if MODE == "walk":
            s = (
                out_df[out_df["split"] == "test"]
                .groupby(["symbol", "model"], as_index=False)
                .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), r2_mean=("r2", "mean"))
                .sort_values(["symbol", "rmse_mean"])
            )
            print("\n[SUMMARY] Walk-forward TEST (mean/std across folds):")
            print(s.to_string(index=False))

    else:
        if not all_clf:
            raise RuntimeError("No results produced. Check data size/splits.")
        out_df = pd.DataFrame([asdict(r) for r in all_clf])
        out_df.to_csv(OUT_PATH, index=False)

        if MODE == "walk":
            s = (
                out_df[out_df["split"] == "test"]
                .groupby(["symbol", "model"], as_index=False)
                .agg(auc_mean=("auc", "mean"), auc_std=("auc", "std"), balacc_mean=("balacc", "mean"))
                .sort_values(["symbol", "auc_mean"], ascending=[True, False])
            )
            print("\n[SUMMARY] Walk-forward TEST (mean/std across folds):")
            print(s.to_string(index=False))

    print("\n[DONE] Finished.")
    print(f"[ARTIFACT] Saved metrics to: {OUT_PATH}")


if __name__ == "__main__":
    main()