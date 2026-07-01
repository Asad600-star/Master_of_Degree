"""
Production-rule extraction from a trained decision tree.
========================================================

A data-grounded replacement for random generation: rules are extracted from the
root-to-leaf paths of a CART decision tree trained on the 56 causal features
across 8 instruments. Each rule corresponds to a real branch of the tree and is
therefore grounded in the data.

Target — three classes by 5-day forward return:
    a1 (Strong Bullish):  r_{t+5} > +tau
    a3 (Bearish):         r_{t+5} < -tau
    a2 (Consider/neutral): otherwise

Reproducibility: random_state = 42.

Run:
    python -m jobs.extract_rules_from_tree
Output: production_rules_knowledge_base.json
"""
from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

# Reuse the SAME 56-feature pipeline as the production models (train_baseline),
# so the rule tree operates on exactly the canonical 56-dimensional feature vector.
from jobs.train_baseline import add_technical_features, build_feature_matrix

SEED = 42
END_DATE = os.environ.get("END_DATE", "2026-06-01")
HORIZON = 5
TAU = 0.01  # +/-1% threshold for classes a1/a3
MAX_DEPTH = int(os.environ.get("TREE_MAX_DEPTH", "7"))
MIN_LEAF = int(os.environ.get("TREE_MIN_LEAF", "80"))
# Chronological split date: rows on/before TRAIN_END train the tree (and the
# knowledge base); later rows form the held-out test set for the ROC/PR figures.
TRAIN_END = os.environ.get("RULES_TRAIN_END", "2023-12-31")
SYMBOLS = ["AAPL", "TSLA", "MSFT", "GLD", "^GSPC", "^IXIC", "^DJI", "^RUT"]

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = Path(os.environ.get("FIG_DIR", "/tmp/figs"))

# Columns read from features_daily (identical to the production inference path).
_FEATURES_SQL = text("""
    SELECT symbol, date,
           open, high, low, close, volume,
           return_1d, log_return,
           sma_5, volatility_5, sma_10, volatility_10, sma_20, volatility_20,
           return_lag_1, return_lag_2, return_lag_3, return_lag_4, return_lag_5,
           mkt_return_1d, mkt_log_return, mkt_mom_5, mkt_mom_10, mkt_mom_20, mkt_vol_20,
           vix_level, vix_return_1d, vix_change_1d,
           irx_level, irx_change_1d,
           tnx_level, tnx_change_1d
    FROM features_daily
    WHERE symbol = :s
    ORDER BY date ASC
""")


def get_engine():
    cfg = dotenv_values(ROOT / ".env")
    url = os.environ.get("DATABASE_URL") or cfg.get("DATABASE_URL")
    return create_engine(url, pool_pre_ping=True)


def load_symbol(engine, sym: str) -> pd.DataFrame:
    df = pd.read_sql(_FEATURES_SQL, engine, params={"s": sym}, parse_dates=["date"])
    df = df[df["date"] <= pd.to_datetime(END_DATE)].reset_index(drop=True)
    return df


def build_dataset(engine) -> tuple[pd.DataFrame, list[str]]:
    """Builds a pooled dataset from the 8 instruments using the canonical
    56-feature matrix (build_feature_matrix) plus a 3-class target."""
    frames = []
    feat_cols = None
    for s in SYMBOLS:
        df = load_symbol(engine, s)
        if len(df) < 300:
            continue
        df = add_technical_features(df)
        df, cols = build_feature_matrix(df)   # canonical 56 features
        if df.empty:
            continue
        # 5-day forward return (target source; dropped from the feature set)
        df["ret_fwd"] = df["close"].shift(-HORIZON) / df["close"] - 1.0
        df = df.dropna(subset=["ret_fwd"]).reset_index(drop=True)
        if feat_cols is None:
            feat_cols = cols
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    # 3-class target
    def lab(r):
        if r > TAU: return "a1"
        if r < -TAU: return "a3"
        return "a2"
    data["target"] = data["ret_fwd"].apply(lab)
    return data, feat_cols


def tree_to_rules(tree: DecisionTreeClassifier, feat_names: list[str]) -> list[dict]:
    """Extracts rules from the tree paths (root -> leaf)."""
    t = tree.tree_
    classes = tree.classes_
    rules = []

    def recurse(node, conds):
        if t.feature[node] != _tree.TREE_UNDEFINED:
            name = feat_names[t.feature[node]]
            thr = t.threshold[node]
            # left branch: feature <= thr
            recurse(t.children_left[node], conds + [(name, "<=", thr)])
            # right branch: feature > thr
            recurse(t.children_right[node], conds + [(name, ">", thr)])
        else:
            # leaf: majority class
            counts = t.value[node][0]
            cls = classes[int(np.argmax(counts))]
            support = int(t.n_node_samples[node])   # number of training samples in the leaf
            purity = float(counts.max() / counts.sum()) if counts.sum() > 0 else 0.0
            rules.append({
                "conditions": conds,
                "consequent": cls,
                "support": support,
                "purity": round(purity, 3),
            })

    recurse(0, [])
    return rules


def main():
    print("=" * 80)
    print(" PRODUCTION-RULE EXTRACTION FROM DECISION TREE")
    print("=" * 80)
    print(f" seed={SEED}  END_DATE={END_DATE}  max_depth={MAX_DEPTH}  min_leaf={MIN_LEAF}  tau=+/-{TAU}")

    engine = get_engine()
    data, feat_cols = build_dataset(engine)

    # Chronological split: the tree (and therefore the knowledge base) is fit on
    # the training block only; the later block is held out for the ROC/PR figures.
    train = data[data["date"] <= pd.to_datetime(TRAIN_END)].reset_index(drop=True)
    test = data[data["date"] > pd.to_datetime(TRAIN_END)].reset_index(drop=True)
    X = train[feat_cols].values
    y = train["target"].values
    print(f" Dataset: {len(data)} rows ({len(feat_cols)} features); "
          f"train={len(train)} (<= {TRAIN_END}), test={len(test)}")
    print(f" Train classes: {dict(pd.Series(y).value_counts())}")

    clf = DecisionTreeClassifier(
        max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF,
        criterion="gini", random_state=SEED,
    )
    clf.fit(X, y)
    acc = clf.score(X, y)
    print(f" Tree trained: train accuracy={acc:.3f}, leaves={clf.get_n_leaves()}")

    # ---- Held-out ROC / PR of the rule system (Buy = class a1) ----------------
    X_te = test[feat_cols].values
    a1_idx = list(clf.classes_).index("a1")
    p_buy = clf.predict_proba(X_te)[:, a1_idx]
    y_buy = (test["ret_fwd"].values > TAU).astype(int)   # actual "strong-bullish" outcome
    test_auc = float(roc_auc_score(y_buy, p_buy))
    print(f" Held-out rule-system ROC AUC (Buy vs rest): {test_auc:.3f} on {len(test)} samples")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Fig 8: ROC
    fpr, tpr, _ = roc_curve(y_buy, p_buy)
    plt.figure(figsize=(5.2, 5.0))
    plt.plot(fpr, tpr, color="#d62728", lw=2.0,
             label=f"Production-rule system (AUC = {test_auc:.3f})")
    plt.fill_between(fpr, fpr, tpr, color="#d62728", alpha=0.10)
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1.0, label="Random (AUC = 0.5)")
    plt.xlabel("False-Positive Rate (1 − Specificity)")
    plt.ylabel("True-Positive Rate (Sensitivity)")
    plt.title("Fig. 8. ROC curve of the production-rule decision system")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig8_roc_rules.png", dpi=150, facecolor="#f7f8fa")
    plt.close()
    # Fig 9: Precision-Recall
    prec, rec, _ = precision_recall_curve(y_buy, p_buy)
    base = y_buy.mean()
    plt.figure(figsize=(5.2, 5.0))
    plt.plot(rec, prec, color="#2ca02c", lw=2.0, label="Production-rule system")
    plt.fill_between(rec, base, prec, color="#2ca02c", alpha=0.10)
    plt.axhline(base, ls="--", color="gray", lw=1.0,
                label=f"Random precision (≈ {base:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Fig. 9. Precision–Recall curve of the production-rule decision system")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig9_pr_rules.png", dpi=150, facecolor="#f7f8fa")
    plt.close()
    print(f" Saved figures -> {FIG_DIR}/fig8_roc_rules.png, fig9_pr_rules.png")

    rules_raw = tree_to_rules(clf, feat_cols)
    print(f" Extracted rules (= number of leaves): {len(rules_raw)}")

    # Convert to p_i format with f-indices
    fidx = {nm: (i + 1) for i, nm in enumerate(feat_cols)}
    def simplify_conditions(conditions):
        """Collapses repeated splits on one feature into a minimal interval.
        A tree path may bound a feature several times; we keep the tightest
        lower (>) and upper (<=) bounds."""
        bounds = {}  # name -> [lower, upper]
        for nm, op, thr in conditions:
            lo, hi = bounds.get(nm, [None, None])
            if op == ">":
                lo = thr if lo is None else max(lo, thr)
            else:  # "<="
                hi = thr if hi is None else min(hi, thr)
            bounds[nm] = [lo, hi]
        parts = []
        for nm in sorted(bounds):
            lo, hi = bounds[nm]
            if lo is not None and hi is not None:
                parts.append(f"{lo:.3f} < {nm} ≤ {hi:.3f}")
            elif lo is not None:
                parts.append(f"{nm} > {lo:.3f}")
            else:
                parts.append(f"{nm} ≤ {hi:.3f}")
        return parts, sorted(bounds.keys())

    rules = []
    for k, r in enumerate(rules_raw, 1):
        parts, ante_names = simplify_conditions(r["conditions"])
        ante = sorted({fidx[nm] for nm in ante_names})
        cond = " ∧ ".join(parts)
        rules.append({
            "id": f"p{k}",
            "antecedent_features": ante_names,
            "ante": ante,
            "cons": r["consequent"],
            "condition": cond,
            "support": r["support"],
            "purity": r["purity"],
        })

    EVENTS = {"a1": "Strong Bullish", "a2": "Consider / neutral", "a3": "Bearish"}
    out = {
        "generated_at": datetime.now().isoformat(),
        "method": "decision-tree path extraction (CART)",
        "seed": SEED,
        "tree_max_depth": MAX_DEPTH,
        "tree_min_samples_leaf": MIN_LEAF,
        "train_accuracy": round(acc, 3),
        "train_end": TRAIN_END,
        "test_size": int(len(test)),
        "test_roc_auc": round(test_auc, 3),
        "total_rules": len(rules),
        "features_count": len(feat_cols),
        "events": EVENTS,
        "rules": rules,
    }
    out_path = ROOT / "production_rules_knowledge_base.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Summary
    cls_counts = pd.Series([r["cons"] for r in rules]).value_counts()
    print(f"\n Rules by class: {dict(cls_counts)}")
    print(f" Mean purity: {np.mean([r['purity'] for r in rules]):.3f}")
    print("\n Sample rules (top-5 by support):")
    for r in sorted(rules, key=lambda x: -x["support"])[:5]:
        ante = " ∧ ".join(f"f{i}" for i in r["ante"])
        print(f"   {r['id']}: {ante} → {r['cons']} (support={r['support']}, purity={r['purity']})")
    print(f"\n[OK] Saved: {out_path} ({len(rules)} rules)")


if __name__ == "__main__":
    main()
