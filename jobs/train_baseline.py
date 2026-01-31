import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

load_dotenv()

TARGET = "target_return_1d"

FEATURE_COLS = [
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
]


def get_env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name, default)
    if v is None or str(v).strip() == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return str(v).strip()


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true >= 0) == (y_pred >= 0)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "diracc": directional_accuracy(y_true, y_pred),
        "posrate": float(np.mean(y_true >= 0)),
    }


def print_eval(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    m = evaluate_metrics(y_true, y_pred)
    print(
        f"[{name}] MAE={m['mae']:.6f} RMSE={m['rmse']:.6f} R2={m['r2']:.4f} "
        f"DirAcc={m['diracc']:.4f} PosRate={m['posrate']:.4f}"
    )
    return m


@dataclass
class RunRow:
    symbol: str
    split: str
    model: str
    n_rows: int
    mae: float
    rmse: float
    r2: float
    diracc: float
    posrate: float


def pick_best_on_val(candidates: list[tuple[str, object]], X_train, y_train, X_val, y_val) -> tuple[str, object]:
    best_name = ""
    best_model = None
    best_rmse = float("inf")

    for name, model in candidates:
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        cur = rmse(y_val, pred)
        if cur < best_rmse:
            best_rmse = cur
            best_name = name
            best_model = model

    assert best_model is not None
    return best_name, best_model


def main() -> None:
    db_url = get_env("DATABASE_URL")

    train_end = get_env("TRAIN_END", "2022-12-31")
    val_end = get_env("VAL_END", "2024-12-31")

    only_symbol = os.environ.get("ONLY_SYMBOL")
    only_symbol = only_symbol.strip() if only_symbol else None

    out_path = Path(os.environ.get("BASELINE_OUT", "artifacts/baseline_metrics.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    print(f"[INFO] Symbols={symbols}")
    print(f"[INFO] Split: train_end={train_end}, val_end={val_end}")

    ridge_candidates: list[tuple[str, object]] = []
    for a in [0.1, 0.3, 1.0, 3.0, 10.0]:
        ridge_candidates.append(
            (
                f"RIDGE(alpha={a})",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", Ridge(alpha=a, random_state=42)),
                    ]
                ),
            )
        )

    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    hgb = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        max_iter=600,
        random_state=42,
    )

    results: list[RunRow] = []

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
                       return_lag_1, return_lag_2, return_lag_3, return_lag_4, return_lag_5,
                       target_return_1d
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

        train = df[df["date"] <= train_end]
        val = df[(df["date"] > train_end) & (df["date"] <= val_end)]
        test = df[df["date"] > val_end]

        if len(train) < 200 or len(val) < 50 or len(test) < 50:
            print(
                f"[WARN] {sym}: not enough rows after split: train={len(train)} val={len(val)} test={len(test)}"
            )
            continue

        X_train, y_train = train[FEATURE_COLS].values, train[TARGET].values
        X_val, y_val = val[FEATURE_COLS].values, val[TARGET].values
        X_test, y_test = test[FEATURE_COLS].values, test[TARGET].values

        print(f"\n=== {sym} ===")
        print(f"[ROWS] train={len(train)} val={len(val)} test={len(test)}")

        # Baselines
        pred0_val = np.zeros_like(y_val)
        pred0_test = np.zeros_like(y_test)

        mean_train = float(np.mean(y_train))
        pred_mean_val = np.full_like(y_val, fill_value=mean_train, dtype=float)
        pred_mean_test = np.full_like(y_test, fill_value=mean_train, dtype=float)

        pred_naive_val = val["return_1d"].values
        pred_naive_test = test["return_1d"].values

        m = print_eval("BASELINE_ZERO(val)", y_val, pred0_val)
        results.append(RunRow(sym, "val", "BASELINE_ZERO", len(val), **m))
        m = print_eval("BASELINE_ZERO(test)", y_test, pred0_test)
        results.append(RunRow(sym, "test", "BASELINE_ZERO", len(test), **m))

        m = print_eval("BASELINE_MEAN_TRAIN(val)", y_val, pred_mean_val)
        results.append(RunRow(sym, "val", "BASELINE_MEAN_TRAIN", len(val), **m))
        m = print_eval("BASELINE_MEAN_TRAIN(test)", y_test, pred_mean_test)
        results.append(RunRow(sym, "test", "BASELINE_MEAN_TRAIN", len(test), **m))

        m = print_eval("BASELINE_NAIVE_RETURN1D(val)", y_val, pred_naive_val)
        results.append(RunRow(sym, "val", "BASELINE_NAIVE_RETURN1D", len(val), **m))
        m = print_eval("BASELINE_NAIVE_RETURN1D(test)", y_test, pred_naive_test)
        results.append(RunRow(sym, "test", "BASELINE_NAIVE_RETURN1D", len(test), **m))

        # Ridge: choose alpha on val
        best_ridge_name, best_ridge = pick_best_on_val(
            ridge_candidates, X_train, y_train, X_val, y_val
        )
        pred_val = best_ridge.predict(X_val)
        pred_test = best_ridge.predict(X_test)
        m = print_eval(f"{best_ridge_name}(val)", y_val, pred_val)
        results.append(RunRow(sym, "val", best_ridge_name, len(val), **m))
        m = print_eval(f"{best_ridge_name}(test)", y_test, pred_test)
        results.append(RunRow(sym, "test", best_ridge_name, len(test), **m))

        rf.fit(X_train, y_train)
        pred_val = rf.predict(X_val)
        pred_test = rf.predict(X_test)
        m = print_eval("RF(val)", y_val, pred_val)
        results.append(RunRow(sym, "val", "RF", len(val), **m))
        m = print_eval("RF(test)", y_test, pred_test)
        results.append(RunRow(sym, "test", "RF", len(test), **m))

        hgb.fit(X_train, y_train)
        pred_val = hgb.predict(X_val)
        pred_test = hgb.predict(X_test)
        m = print_eval("HGB(val)", y_val, pred_val)
        results.append(RunRow(sym, "val", "HGB", len(val), **m))
        m = print_eval("HGB(test)", y_test, pred_test)
        results.append(RunRow(sym, "test", "HGB", len(test), **m))

    if not results:
        raise RuntimeError("No results produced. Check features_daily size/splits.")

    out_df = pd.DataFrame([asdict(r) for r in results])
    out_df = out_df.sort_values(
        ["symbol", "split", "rmse", "mae"], ascending=[True, True, True, True]
    )
    out_df.to_csv(out_path, index=False)

    print("\n[DONE] Baseline training finished.")
    print(f"[ARTIFACT] Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()