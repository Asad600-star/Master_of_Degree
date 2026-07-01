"""
RACC — Risk-Aware Conformal Control for the production-rule decision layer.
==========================================================================

A risk-controlled, abstaining decision layer that unifies the system's two
forecasts and gives a distribution-free guarantee on the *monetary* risk of the
trades it chooses to take (rather than on classification error).

Pipeline (all on the canonical 56 features, pooled over the 8 instruments,
chronological train / calibration / test split):

  1. Calibrated direction probability  p_cal = PAV( LogReg P(up) ).
  2. Production-rule belief             q_up  = weighted aggregation of fired
                                                CART rules (purity * log(1+support));
                                                for the single-tree layer this is the
                                                leaf class-probability for a1 (Buy).
  3. Fusion                             pi = lambda * p_cal + (1 - lambda) * q_up.
  4. Volatility-scaled loss            l_i = clip( max(0, -d_i * r_i) / sigma_hat_i, 0, B ),
                                                d_i = sign(pi_i - 0.5), only when acting.
  5. Conformal Risk Control (Angelopoulos et al., 2023): choose the abstention
     threshold t so that  E[L_test] <= alpha  holds distribution-free.

The system ACTS (Buy/Sell) only when |pi - 0.5| >= t and ABSTAINS (no-trade,
i.e. the rule action a2) otherwise.

Run:
    python -m jobs.racc
Outputs:
    artifacts/racc_results.csv, artifacts/racc_aggregation.csv
    /tmp/figs/fig12_risk_control.png, /tmp/figs/fig13_coverage_accuracy.png
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
from sklearn.metrics import roc_auc_score, accuracy_score

from jobs.train_baseline import add_technical_features, build_feature_matrix, compute_targets

load_dotenv()

SEED = 42
END_DATE = os.environ.get("END_DATE", "2026-06-01")
HORIZON = 5
TAU = 0.01                      # +/-1% threshold for the a1/a2/a3 rule classes
TRAIN_END = "2022-12-31"        # <= : train the models
CALIB_END = "2023-12-31"        # (TRAIN_END, CALIB_END] : conformal calibration
                                # > CALIB_END : held-out test
B = 3.0                          # loss bound (volatility-normalised downside)
EPS = 1e-4
SYMBOLS = ["AAPL", "TSLA", "MSFT", "GLD", "^GSPC", "^IXIC", "^DJI", "^RUT"]

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts"
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
    """Pooled dataset on the canonical 56 features + the targets RACC needs."""
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
        df = compute_targets(df, HORIZON)            # target_return_kd, target_direction, target_vol_kd
        feat = feat or cols
        frames.append(df)
    data = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    # 3-class rule target
    r = data["target_return_kd"].values
    data["cls3"] = np.where(r > TAU, 0, np.where(r < -TAU, 2, 1))   # 0=a1 up,1=a2,2=a3 down
    return data, feat


def split(data):
    tr = data[data["date"] <= pd.to_datetime(TRAIN_END)]
    ca = data[(data["date"] > pd.to_datetime(TRAIN_END)) & (data["date"] <= pd.to_datetime(CALIB_END))]
    te = data[data["date"] > pd.to_datetime(CALIB_END)]
    return tr.reset_index(drop=True), ca.reset_index(drop=True), te.reset_index(drop=True)


def crc_threshold(s_abs, loss, alpha, grid):
    """Conformal Risk Control: smallest threshold t with the finite-sample
    upper bound on the risk <= alpha. Loss is non-increasing in t."""
    n = len(loss)
    best_t = grid[-1]
    for t in grid:                                   # ascending -> first feasible = smallest t
        risk_hat = np.mean(loss * (s_abs >= t))
        bound = (n * risk_hat + B) / (n + 1)
        if bound <= alpha:
            best_t = t
            break
    return best_t


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    engine = get_engine()
    data, feat = build_pool(engine)
    tr, ca, te = split(data)
    print(f"pool={len(data)}  train={len(tr)}  calib={len(ca)}  test={len(te)}  features={len(feat)}")

    Xtr, Xca, Xte = tr[feat].values, ca[feat].values, te[feat].values

    # --- direction probability (PAV-calibrated LogReg) ---
    clf = Pipeline([("sc", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced",
                                              random_state=SEED))])
    clf.fit(Xtr, tr["target_direction"].values)
    praw_ca = clf.predict_proba(Xca)[:, 1]
    praw_te = clf.predict_proba(Xte)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(praw_ca, ca["target_direction"].values)
    pcal_ca, pcal_te = iso.predict(praw_ca), iso.predict(praw_te)

    # --- production-rule belief from the CART rule tree (q_up = P(a1)) ---
    tree = DecisionTreeClassifier(max_depth=7, min_samples_leaf=80, random_state=SEED).fit(Xtr, tr["cls3"].values)
    a1 = list(tree.classes_).index(0)
    qw_ca, qw_te = tree.predict_proba(Xca)[:, a1], tree.predict_proba(Xte)[:, a1]   # weighted (leaf purity)
    leaf_cls = tree.classes_[np.argmax(tree.predict_proba(Xte), axis=1)]
    qm_te = (leaf_cls == 0).astype(float)                                          # majority (hard)

    # --- volatility forecast sigma_hat (ExtraTrees) ---
    et = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=10,
                             random_state=SEED, n_jobs=-1).fit(Xtr, tr["target_vol_kd"].values)
    sig_ca, sig_te = et.predict(Xca), et.predict(Xte)

    # --- fusion weight lambda tuned on calibration (maximise AUC of pi for up) ---
    yca = ca["target_direction"].values
    best_lam, best_auc = 0.5, -1
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        pi = lam * pcal_ca + (1 - lam) * qw_ca
        try:
            au = roc_auc_score(yca, pi)
        except Exception:
            au = 0.5
        if au > best_auc:
            best_auc, best_lam = au, lam
    lam = best_lam
    print(f"fusion lambda={lam} (calib AUC={best_auc:.3f})")

    def decision_arrays(pcal, qw, sig, part):
        pi = lam * pcal + (1 - lam) * qw
        s = pi - 0.5
        d = np.sign(s); d[d == 0] = 1
        r = part["target_return_kd"].values
        loss = np.clip(np.maximum(0.0, -d * r) / np.maximum(sig, EPS), 0, B)
        up_true = part["target_direction"].values
        return s, d, r, loss, up_true

    s_ca, d_ca, r_ca, loss_ca, _ = decision_arrays(pcal_ca, qw_ca, sig_ca, ca)
    s_te, d_te, r_te, loss_te, up_te = decision_arrays(pcal_te, qw_te, sig_te, te)
    grid = np.quantile(np.abs(s_ca), np.linspace(0, 1, 101))

    # --- sweep target risk alpha: CRC on calib, evaluate on test ---
    rows = []
    for alpha in np.round(np.arange(0.05, 0.81, 0.05), 2):
        t = crc_threshold(np.abs(s_ca), loss_ca, alpha, grid)
        act = np.abs(s_te) >= t
        cov = float(np.mean(act))
        realized = float(np.mean(loss_te * act))
        # directional accuracy on the acted subset (predicted up/down vs realised)
        if act.sum() > 0:
            pred_up = (d_te[act] > 0).astype(int)
            acc = float(accuracy_score(up_te[act], pred_up))
        else:
            acc = float("nan")
        rows.append(dict(alpha=alpha, threshold=round(float(t), 4), coverage=round(cov, 3),
                         realized_risk=round(realized, 4), guarantee_holds=bool(realized <= alpha),
                         acc_acted=round(acc, 3) if act.sum() else None))
    res = pd.DataFrame(rows)
    res.to_csv(A / "racc_results.csv", index=False)
    acc_all = accuracy_score(up_te, (d_te > 0).astype(int))
    print(res.to_string(index=False))
    print(f"\nUnconditional directional accuracy (act always): {acc_all:.3f}")
    print(f"Guarantee held on test for {res.guarantee_holds.mean()*100:.0f}% of alpha levels")

    # --- weighted vs majority rule aggregation (selective AUC of pi for up) ---
    agg = []
    for name, q in [("weighted", qw_te), ("majority", qm_te)]:
        pi = lam * pcal_te + (1 - lam) * q
        try:
            au = roc_auc_score(up_te, pi)
        except Exception:
            au = 0.5
        agg.append(dict(aggregation=name, fused_auc=round(float(au), 4)))
    pd.DataFrame(agg).to_csv(A / "racc_aggregation.csv", index=False)
    print("\nRule aggregation (fused AUC):", agg)

    # --- Fig 12: risk-control curve (target vs realised; must lie on/below diagonal) ---
    plt.figure(figsize=(5.6, 5.2))
    plt.plot([0, 0.8], [0, 0.8], "--", color="gray", lw=1, label="target = realised")
    plt.plot(res.alpha, res.realized_risk, "o-", color="#1f77b4", lw=1.8, label="RACC realised risk")
    plt.fill_between(res.alpha, 0, res.alpha, color="#2ca02c", alpha=0.08, label="guarantee satisfied")
    plt.xlabel("Target risk level α (volatility-scaled loss)")
    plt.ylabel("Realised risk on held-out test")
    plt.title("Fig. 12. RACC risk-control guarantee")
    plt.legend(fontsize=8, loc="upper left"); plt.tight_layout()
    plt.savefig(FIG / "fig12_risk_control.png", dpi=150, facecolor="white"); plt.close()

    # --- Fig 13: risk-coverage frontier (monotone; the dialable risk budget) ---
    rc = res.sort_values("coverage")
    fig, ax1 = plt.subplots(figsize=(6.0, 5.0))
    ax1.plot(rc.coverage, rc.realized_risk, "o-", color="#1f77b4", lw=1.9)
    for _, row in rc.iterrows():
        if row["alpha"] in (0.05, 0.30, 0.60, 0.65):
            ax1.annotate(f"α={row['alpha']:.2f}", (row["coverage"], row["realized_risk"]),
                         textcoords="offset points", xytext=(6, -10), fontsize=7.5, color="#444")
    ax1.set_xlabel("Coverage (fraction of days the system trades)")
    ax1.set_ylabel("Realised volatility-scaled risk on test")
    ax1.set_title("Fig. 13. RACC risk–coverage frontier")
    ax1.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(FIG / "fig13_coverage_accuracy.png", dpi=150, facecolor="white"); plt.close()

    print(f"\n[ARTIFACT] {A/'racc_results.csv'}, {A/'racc_aggregation.csv'}")
    print(f"[FIG] {FIG/'fig12_risk_control.png'}, {FIG/'fig13_coverage_accuracy.png'}")


if __name__ == "__main__":
    main()
