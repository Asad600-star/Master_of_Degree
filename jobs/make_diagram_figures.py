"""
Regenerates the two diagram figures used in the paper.
========================================================

- Fig. 1: end-to-end system architecture (eight modules).
- Fig. 4: illustrative multi-instrument daily snapshot of the live system
          (eight instruments, 5-day horizon, 22 walk-forward folds).

These are diagram/illustrative figures (not data plots); the snapshot values in
Fig. 4 are representative of a single trading day. Run:

    python -m jobs.make_diagram_figures

Output: /tmp/figs/fig1_architecture.png and /tmp/figs/fig4_snapshot.png
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = Path(os.environ.get("FIG_DIR", "/tmp/figs"))
N_RULES = 69          # current production-rule count (decision-tree extraction)
N_FEATURES = 56
N_FOLDS = 22

BG = "#f7f8fa"
INK = "#1f2d3d"


# ─────────────────────────────────────────────────────────────────────
#  Fig. 1 — architecture
# ─────────────────────────────────────────────────────────────────────
def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("Fig. 1. End-to-end system architecture", fontsize=13,
                 fontweight="bold", pad=12)

    boxes = [
        # (x, y, text, facecolor)
        (0.6, 5.6, "Data Ingest\n(yfinance API)\nPostgreSQL", "#a8d5a2"),
        (4.4, 5.6, f"Feature\nEngineering\n|F| = {N_FEATURES}", "#9ec5e8"),
        (8.2, 5.6, "Hybrid Models\nVOTING + STACK\n+ ExtraTrees", "#f5c98a"),
        (12.0, 5.6, "Isotonic\nCalibration\n(PAV)", "#c9a8e0"),
        (0.6, 1.4, f"Production\nRules ({N_RULES})\nP : F → A", "#f3a6a6"),
        (4.4, 1.4, "Risk Manager\nVaR / Sharpe /\nPosition Size", "#f2d98a"),
        (8.2, 1.4, "SHAP\nExplainer\n(Top-15)", "#a8d5a2"),
        (12.0, 1.4, "User Interfaces\nStreamlit +\nTelegram Bot", "#b6b6e8"),
    ]
    bw, bh = 3.4, 2.0
    centers = []
    for x, y, text, fc in boxes:
        ax.add_patch(FancyBboxPatch((x, y), bw, bh,
                     boxstyle="round,pad=0.04,rounding_size=0.18",
                     fc=fc, ec="#5a6b7b", lw=1.2))
        ax.text(x + bw / 2, y + bh / 2, text, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=INK)
        centers.append((x + bw / 2, y, x, y + bh / 2, x + bw, y + bh / 2))

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14,
                     lw=1.4, color="#5a6b7b"))

    # top row left -> right
    for i in range(3):
        arrow((boxes[i][0] + bw, boxes[i][1] + bh / 2),
              (boxes[i + 1][0], boxes[i + 1][1] + bh / 2))
    # bottom row left -> right
    for i in range(4, 7):
        arrow((boxes[i][0] + bw, boxes[i][1] + bh / 2),
              (boxes[i + 1][0], boxes[i + 1][1] + bh / 2))
    # data-ingest -> production-rules (vertical)
    arrow((boxes[0][0] + bw / 2, boxes[0][1]),
          (boxes[4][0] + bw / 2, boxes[4][1] + bh))

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig1_architecture.png", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[OK] {FIG_DIR}/fig1_architecture.png")


# ─────────────────────────────────────────────────────────────────────
#  Fig. 4 — illustrative multi-instrument snapshot (8 instruments)
# ─────────────────────────────────────────────────────────────────────
def fig_snapshot() -> None:
    # One representative trading day; values are illustrative but internally
    # consistent (recommendation follows the calibrated up-probability).
    cards = [
        # ticker, name, p_up, sigma%, var%, sharpe, position%
        ("AAPL",  "Apple Inc.",        0.642, 1.48, -5.45,  1.84, 10),
        ("TSLA",  "Tesla Inc.",        0.574, 2.18, -9.55,  0.92,  5),
        ("MSFT",  "Microsoft Corp.",   0.611, 1.36, -4.98,  1.51,  9),
        ("GLD",   "SPDR Gold Trust",   0.558, 0.84, -2.91,  0.74,  4),
        ("^GSPC", "S&P 500",           0.492, 0.71, -2.61, -0.41,  0),
        ("^IXIC", "NASDAQ Comp.",      0.560, 1.23, -4.52,  0.66,  6),
        ("^DJI",  "Dow Jones",         0.470, 0.68, -2.40, -0.55,  0),
        ("^RUT",  "Russell 2000",      0.515, 1.02, -3.74,  0.18,  2),
    ]

    def verdict(p):
        if p >= 0.58:
            return "Buy", "#2e9e5b"
        if p >= 0.52:
            return "Consider", "#e08a1e"
        return "Do not buy", "#cf4040"

    fig, axes = plt.subplots(2, 4, figsize=(13.0, 9.0))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Fig. 4. Illustrative daily snapshot of the live system "
                 "(8 instruments, 5-day horizon)",
                 fontsize=12.5, fontweight="bold", y=0.99)

    for ax, c in zip(axes.ravel(), cards):
        ticker, name, p, sig, var, sharpe, pos = c
        label, col = verdict(p)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                     boxstyle="round,pad=0.01,rounding_size=0.03",
                     transform=ax.transAxes, fc="white", ec=col, lw=2.2))
        # header bar
        ax.add_patch(FancyBboxPatch((0.02, 0.80), 0.96, 0.18,
                     boxstyle="round,pad=0.01,rounding_size=0.03",
                     transform=ax.transAxes, fc=col, ec=col, lw=0))
        ax.text(0.07, 0.89, ticker, transform=ax.transAxes, fontsize=14,
                fontweight="bold", color="white", va="center")
        ax.text(0.07, 0.835, name, transform=ax.transAxes, fontsize=8,
                color="white", va="center")
        # recommendation
        ax.text(0.5, 0.70, label, transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=col, ha="center")
        # probability
        ax.text(0.5, 0.55, f"{p:.3f}", transform=ax.transAxes, fontsize=20,
                fontweight="bold", color=INK, ha="center")
        ax.text(0.5, 0.46, "p̂ — prob. up move (5d)", transform=ax.transAxes,
                fontsize=7.5, color="#6b7280", ha="center")
        # risk row
        ax.text(0.5, 0.32,
                f"σ̂ 5d: {sig:.2f}%   VaR₉₅: {var:.2f}%   Sharpe: {sharpe:+.2f}",
                transform=ax.transAxes, fontsize=8.5, color=INK, ha="center")
        # position
        ax.text(0.5, 0.18, f"Position size: {pos}%", transform=ax.transAxes,
                fontsize=9, fontweight="bold", color=col, ha="center")

    footer = (f"Pipeline: HYBRID_VOTING (LR+HGB+XGB+LGBM+RF, soft-vote) → PAV-calibrated p̂  |  "
              f"Volatility: ExtraTrees (N=500, max_depth=12, min_samples_leaf=10)  |  "
              f"Risk: VaR₅d = σ̂·√5, asymmetric z ∈ {{1.645, 1.96, 2.33}}  |  "
              f"{N_FEATURES}-D causal features → {N_RULES} production rules p : F → A  |  "
              f"K = {N_FOLDS} walk-forward folds")
    fig.text(0.5, 0.02, footer, ha="center", fontsize=7.6, color="#34435a", wrap=True)

    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig4_snapshot.png", dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"[OK] {FIG_DIR}/fig4_snapshot.png")


def main() -> None:
    fig_architecture()
    fig_snapshot()


if __name__ == "__main__":
    main()
