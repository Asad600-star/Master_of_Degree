"""
RACC-gated economic back-test.
==============================

Tests whether abstaining when the conformal risk bound cannot be met (RACC)
produces better *risk-adjusted* trading than acting on every signal.

For each instrument, on the held-out test block (2024-2026), a daily directional
strategy is simulated under three policies:
  - UNGATED   : take the signed signal every day.
  - RACC@alpha: take the signal only when |s| >= t_alpha (else stay in cash).
  - BUY&HOLD  : passive long.

Metrics: annualised Sharpe, max drawdown, total return, coverage (days in market),
number of position changes. Models are fit on data up to 2022; the conformal
threshold is calibrated on 2023 (no test peeking).

Run:
    python -m jobs.racc_backtest
Outputs: artifacts/racc_backtest.csv, /tmp/figs/fig14_racc_equity.png
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesRegressor

from jobs.train_baseline import add_technical_features, build_feature_matrix, compute_targets

load_dotenv()
SEED = 42
END_DATE = os.environ.get("END_DATE", "2026-06-01")
HORIZON = 5
TAU = 0.01
TRAIN_END = "2022-12-31"
CALIB_END = "2023-12-31"
B = 3.0
EPS = 1e-4
FEE = 5e-4                       # 5 bps per position change
ALPHAS = [0.30, 0.45]            # RACC risk budgets to report
SYMBOLS = ["AAPL", "TSLA", "MSFT", "GLD", "^GSPC", "^IXIC", "^DJI", "^RUT"]
LABEL = {"^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^DJI": "Dow Jones", "^RUT": "Russell 2k"}
A = Path(__file__).resolve().parents[1] / "artifacts"
FIG = Path(os.environ.get("FIG_DIR", "/tmp/figs"))

_SQL = text("""
    SELECT symbol, date, open, high, low, close, volume,
           return_1d, log_return,
           sma_5, volatility_5, sma_10, volatility_10, sma_20, volatility_20,
           return_lag_1, return_lag_2, return_lag_3, return_lag_4, return_lag_5,
           mkt_return_1d, mkt_log_return, mkt_mom_5, mkt_mom_10, mkt_mom_20, mkt_vol_20,
           vix_level, vix_return_1d, vix_change_1d,
           irx_level, irx_change_1d, tnx_level, tnx_change_1d
    FROM features_daily WHERE symbol = :s ORDER BY date ASC
""")


def get_engine():
    return create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)


def build_pool(engine):
    frames, feat = [], None
    for s in SYMBOLS:
        df = pd.read_sql(_SQL, engine, params={"s": s}, parse_dates=["date"])
        df = df[df["date"] <= pd.to_datetime(END_DATE)].reset_index(drop=True)
        if len(df) < 400:
            continue
        df = add_technical_features(df)
        df, cols = build_feature_matrix(df)
        if df.empty:
            continue
        keep_ret = df["return_1d"].values.copy()
        df = compute_targets(df, HORIZON)
        feat = feat or cols
        frames.append(df)
    data = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    r = data["target_return_kd"].values
    data["cls3"] = np.where(r > TAU, 0, np.where(r < -TAU, 2, 1))
    return data, feat


def crc_threshold(s_abs, loss, alpha, grid):
    n = len(loss)
    for t in grid:
        if (n * np.mean(loss * (s_abs >= t)) + B) / (n + 1) <= alpha:
            return t
    return grid[-1]


def perf(pnl):
    """Annualised Sharpe, max drawdown, total return from a daily-pnl series."""
    pnl = np.asarray(pnl)
    if pnl.std() < 1e-12:
        sharpe = 0.0
    else:
        sharpe = pnl.mean() / pnl.std() * np.sqrt(252)
    eq = np.cumprod(1.0 + pnl)
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    tot = float(eq[-1] - 1.0)
    return sharpe, mdd, tot, eq


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    engine = get_engine()
    data, feat = build_pool(engine)
    tr = data[data["date"] <= pd.to_datetime(TRAIN_END)]
    ca = data[(data["date"] > pd.to_datetime(TRAIN_END)) & (data["date"] <= pd.to_datetime(CALIB_END))]
    te = data[data["date"] > pd.to_datetime(CALIB_END)]
    print(f"train={len(tr)} calib={len(ca)} test={len(te)}")

    clf = Pipeline([("sc", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced", random_state=SEED))])
    clf.fit(tr[feat].values, tr["target_direction"].values)
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        clf.predict_proba(ca[feat].values)[:, 1], ca["target_direction"].values)
    tree = DecisionTreeClassifier(max_depth=7, min_samples_leaf=80, random_state=SEED).fit(tr[feat].values, tr["cls3"].values)
    a1 = list(tree.classes_).index(0)
    et = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10,
                             random_state=SEED, n_jobs=-1).fit(tr[feat].values, tr["target_vol_kd"].values)

    def signal(part):
        p = iso.predict(clf.predict_proba(part[feat].values)[:, 1])
        q = tree.predict_proba(part[feat].values)[:, a1]
        s = 0.5 * p + 0.5 * q - 0.5          # lambda = 0.5
        sig = et.predict(part[feat].values)
        return s, sig

    # conformal thresholds from calibration
    s_ca, sig_ca = signal(ca)
    d_ca = np.sign(s_ca); d_ca[d_ca == 0] = 1
    r_ca = ca["target_return_kd"].values
    loss_ca = np.clip(np.maximum(0.0, -d_ca * r_ca) / np.maximum(sig_ca, EPS), 0, B)
    grid = np.quantile(np.abs(s_ca), np.linspace(0, 1, 101))
    thr = {al: crc_threshold(np.abs(s_ca), loss_ca, al, grid) for al in ALPHAS}
    print("RACC thresholds:", {a: round(float(t), 4) for a, t in thr.items()})

    # per-instrument daily back-test on test block
    rows = []
    port = {"UNGATED": [], "BUY&HOLD": [], **{f"RACC@{a}": [] for a in ALPHAS}}
    for sym in SYMBOLS:
        d = te[te["symbol"] == sym].sort_values("date")
        if len(d) < 50:
            continue
        s, _ = signal(d)
        ret = d["return_1d"].values
        fwd = np.roll(ret, -1); fwd[-1] = 0.0          # next-day return
        pol = {"UNGATED": np.sign(s)}
        for al in ALPHAS:
            pol[f"RACC@{al}"] = np.sign(s) * (np.abs(s) >= thr[al])
        pol["BUY&HOLD"] = np.ones_like(s)
        for name, pos in pol.items():
            pos = pos.astype(float); pos[pos == 0] = 0.0 if name != "BUY&HOLD" else 1.0
            cost = FEE * np.abs(np.diff(np.concatenate([[0.0], pos])))
            pnl = pos * fwd - cost
            sh, mdd, tot, _ = perf(pnl)
            cov = float(np.mean(pos != 0))
            nch = int(np.sum(np.abs(np.diff(np.concatenate([[0.0], pos]))) > 0))
            rows.append(dict(symbol=LABEL.get(sym, sym), policy=name, sharpe=round(sh, 2),
                             max_drawdown=round(mdd, 3), total_return=round(tot, 3),
                             coverage=round(cov, 2), n_changes=nch))
            port[name].append(pnl)

    res = pd.DataFrame(rows)
    res.to_csv(A / "racc_backtest.csv", index=False)

    # portfolio = equal-weight average daily pnl across instruments
    print("\n=== PORTFOLIO (equal-weight) ===")
    L = min(len(x) for x in port["UNGATED"])
    eqs = {}
    psum = []
    for name, lst in port.items():
        m = np.mean([p[:L] for p in lst], axis=0)
        sh, mdd, tot, eq = perf(m)
        eqs[name] = eq
        cov = np.mean([np.mean(np.asarray(p) != 0) for p in lst]) if name != "BUY&HOLD" else 1.0
        psum.append(dict(policy=name, sharpe=round(sh, 2), max_drawdown=round(mdd, 3),
                         total_return=round(tot, 3), coverage=round(float(cov), 2)))
    pdf = pd.DataFrame(psum)
    print(pdf.to_string(index=False))
    pdf.to_csv(A / "racc_backtest_portfolio.csv", index=False)

    # equity figure
    plt.figure(figsize=(7.2, 4.6))
    colors = {"UNGATED": "#888", "BUY&HOLD": "#999", "RACC@0.3": "#2ca02c", "RACC@0.45": "#1f77b4"}
    style = {"BUY&HOLD": "--"}
    for name, eq in eqs.items():
        plt.plot(eq, style.get(name, "-"), lw=1.9, label=name,
                 color={"UNGATED": "#d62728", "BUY&HOLD": "#777",
                        "RACC@0.3": "#2ca02c", "RACC@0.45": "#1f77b4"}.get(name))
    plt.axhline(1.0, color="gray", lw=0.6)
    plt.xlabel("Trading day (held-out test 2024–2026)")
    plt.ylabel("Equity (start = 1.0)")
    plt.title("Fig. 14. RACC-gated vs ungated portfolio equity")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(FIG / "fig14_racc_equity.png", dpi=150, facecolor="white"); plt.close()
    print(f"\n[ARTIFACT] {A/'racc_backtest.csv'}, {A/'racc_backtest_portfolio.csv'}")
    print(f"[FIG] {FIG/'fig14_racc_equity.png'}")


if __name__ == "__main__":
    main()
