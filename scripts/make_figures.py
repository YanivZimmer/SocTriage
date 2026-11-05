"""Regenerate the final paper's two figures from released CSVs only.

  Figure 1 (default_all_final.pdf): default (argmax) verdicts at full coverage, GUIDE (17 conditions) and
           production (13 methods): malicious caught and non-malicious falsely escalated, as % of class.
  Figure 2 (cost_vs_c_final.pdf/.png): cost curves c*misses + FE at the 80% test-quantile gate, log axes, with the
           main regime c <= 10 shaded and the best local model's crossover with the Decisions API + exemplar dotted.

Inputs:
  --guide       output directory of scripts/analyze_guide.py (default: aggregates/guide, the released snapshot of
                that output); uses full_coverage.csv and frontier_cost.csv
  --production  aggregates/production (default); uses default_verdicts.csv and gate80_test_quantile.csv
Outputs (--out, default figures/).

All in-figure text is 9 pt, as in the paper (AAAI kit minimum). The paper typeset the figures with LaTeX (pgf,
newtx fonts); this script uses matplotlib's bundled STIX serif fonts, so glyphs differ slightly, but every bar,
curve, label and crossover is drawn from the same counts. The crossover labels are the paper's rounded values
(125 and 21); the script checks that the released counts give them (GUIDE 124.7; production 21.5 for RiskFit
rho=0.8 and 21.4 for MaxSim, "about 21" in the paper).

Usage: python scripts/make_figures.py [--guide results/guide] [--out figures]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FULL_W, H_DEFAULT, H_COST = 7.0, 3.05, 2.35  # inches
INK, INK2 = "#0b0b0b", "#52514e"
GRAY, FE_COL, TINY = "#cfcdc6", "#5f5e5a", "#0f8b8d"
C = {"RiskFit": "#2a78d6", "Decisions": "#eb6834", "CE": "#1baf7a", "LightGBM": "#eda100", "Gemini": "#e87ba4",
     "Jev": "#008300", "Opus": "#6250d6", "CyberSecQwen": "#e34948"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#898781", "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
                     "grid.color": "#e1e0d9", "grid.linewidth": 0.5})

# (label in the figure, method name in the CSV)
GUIDE_BARS = [("RiskFit", "RiskFit"), ("CE", "CE"), ("Tiny Decision", "Tiny Decision"), ("LightGBM", "LightGBM"),
              ("MaxSim", "MaxSim"), ("$k$NN", "kNN"), ("Decisions + 20 ex.", "Decisions + 20 exemplars"),
              ("Jev + 20 ex.", "Jev + 20 exemplars"), ("Decisions + 5 ex.", "Decisions + 5 exemplars"),
              ("Decisions + ex.", "Decisions + exemplar"), ("Decisions zero-shot", "Decisions zero-shot"),
              ("Jev + 5 ex.", "Jev + 5 exemplars"), ("Jev + ex.", "Jev + exemplar"), ("Jev zero-shot", "Jev zero-shot"),
              ("Opus 5.5 + ex.", "Opus + exemplar"), ("Opus 5.5 zero-shot", "Opus zero-shot"),
              ("CyberSecQwen-4B", "CyberSecQwen-4B zero-shot")]
PROD_BARS = [("RiskFit", "RiskFit"), (r"RiskFit $\rho{=}0.8$", "RiskFit rho=0.8"), ("CE", "CE"),
             ("Focal-Tversky", "Focal-Tversky"), ("MaxSim", "MaxSim"), ("LightGBM", "LightGBM"),
             ("Tiny Decision", "Tiny Decision"), ("SetFit-LoRA", "SetFit-LoRA"), ("$k$NN", "kNN (frozen)"),
             ("Gemini + ex.", "Gemini + exemplar"), ("Gemini zero-shot", "Gemini zero-shot"),
             ("Decisions + ex.", "Decisions + exemplar"), ("Decisions zero-shot", "Decisions zero-shot")]
GUIDE_SERIES = [("RiskFit", "RiskFit", C["RiskFit"], "o", "-", True), ("CE", "CE", C["CE"], "s", "-", True),
                ("LightGBM", "LightGBM", C["LightGBM"], "^", "-", True),
                ("Decisions + ex.", "Decisions + exemplar", C["Decisions"], "P", "-", True),
                ("Decisions + 5 ex.", "Decisions + 5 exemplars", C["Decisions"], "P", ":", False),
                ("Jev + ex.", "Jev + exemplar", C["Jev"], "X", "-", True),
                ("Opus 5.5 + ex.", "Opus + exemplar", C["Opus"], "v", "-", True),
                ("CyberSecQwen-4B", "CyberSecQwen-4B zero-shot", C["CyberSecQwen"], "*", "--", True)]
PROD_SERIES = [("RiskFit", "RiskFit", C["RiskFit"], "o", "-", True),
               (r"RiskFit $\rho{=}0.8$", "RiskFit rho=0.8", C["RiskFit"], "o", "--", False),
               ("CE", "CE", C["CE"], "s", "-", True), ("MaxSim", "MaxSim", INK2, "D", ":", True),
               ("LightGBM", "LightGBM", C["LightGBM"], "^", "-", True),
               ("Gemini + ex.", "Gemini + exemplar", C["Gemini"], "v", "-", True),
               ("Decisions + ex.", "Decisions + exemplar", C["Decisions"], "P", "-", True),
               ("Decisions zero-shot", "Decisions zero-shot", C["Decisions"], "P", "--", False)]
LOCAL_GUIDE = ("RiskFit", "CE", "LightGBM", "MaxSim", "kNN", "Tiny Decision")
LOCAL_PROD = ("RiskFit", "RiskFit rho=0.8", "RiskFit lambda=4", "CE", "Focal-Tversky", "MaxSim", "kNN (frozen)",
              "LightGBM", "SetFit-LoRA", "Tiny Decision")
CROSS_LABEL = {"GUIDE": 125, "production": 21}  # the paper's rounded crossovers (checked in main())


def rows(p: Path):
    return list(csv.DictReader(p.open(newline="")))


def load(guide: Path, prod: Path):
    gfull = {r["method"]: (int(r["caught"]), int(r["false_mal"])) for r in rows(guide / "full_coverage.csv")
             if r["scope"].startswith("pooled")}
    ggate = {r["method"]: (int(r["misses"]), int(r["false_esc"])) for r in rows(guide / "frontier_cost.csv")}
    pfull = {r["method"]: (int(r["caught"]), int(r["false_esc"])) for r in rows(prod / "default_verdicts.csv") if r["scope"] == "all"}
    pgate = {r["method"]: (int(r["misses"]), int(r["false_esc"])) for r in rows(prod / "gate80_test_quantile.csv") if r["scope"] == "all"}
    return gfull, ggate, pfull, pgate


def best_crossover(gate: dict, local: tuple, hosted: str) -> float:
    """Largest c* below which some local model is cheaper than `hosted` (c * misses + false escalations)."""
    mh, fh = gate[hosted]
    return max((fh - gate[m][1]) / (gate[m][0] - mh) for m in local if m in gate and gate[m][0] > mh)


def color(lab: str) -> str:
    if lab.startswith("Tiny"):
        return TINY
    if lab in ("Focal-Tversky", "MaxSim", "SetFit-LoRA", "$k$NN"):
        return GRAY
    return next((v for k, v in C.items() if lab.startswith(k)), GRAY)


def figure1(gfull, pfull, out: Path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, H_DEFAULT), layout="constrained")
    panels = [(a1, [(lab, *gfull[k]) for lab, k in GUIDE_BARS], 456, 623, "GUIDE (of 456 / 623)"),
              (a2, [(lab, *pfull[k]) for lab, k in PROD_BARS], 73, 1419, "Production (of 73 / 1,419)")]
    for ax, d, nm, nn, title in panels:
        y = np.arange(len(d))[::-1] * 1.0
        h = 0.4
        for yi, (lab, caught, fe) in zip(y, d):
            bold = lab == "RiskFit"
            alpha = 0.55 if lab.startswith("RiskFit") and not bold else 1
            ws = []
            for yy, v, n, c in [(yi + h / 2, caught, nm, color(lab)), (yi - h / 2, fe, nn, FE_COL)]:
                ws.append(v / n * 100)
                ax.barh(yy, ws[-1], h, color=c, alpha=alpha if c != FE_COL else 0.85,
                        edgecolor="#8a8882" if c == GRAY else "none", lw=0.5)
            ax.text(max(ws) + 1.5, yi, f"{caught} / {fe} FE", va="center", ha="left", fontsize=9, color=INK,
                    fontweight="bold" if bold else "normal")
        ax.set_yticks(y)
        ax.set_yticklabels([x[0] for x in d], fontsize=9)
        ax.set_xlim(0, 134)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("% of class")
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(True, axis="x")
    fig.savefig(out / "default_all_final.pdf")
    plt.close(fig)


def figure2(ggate, pgate, out: Path):
    cs = np.logspace(0, np.log10(300), 200)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, H_COST), layout="constrained")
    for ax, data, series, title, cross, ylim in [
            (axes[0], ggate, GUIDE_SERIES, "GUIDE (456 malicious, 623 not)", CROSS_LABEL["GUIDE"], (40, 4e5)),
            (axes[1], pgate, PROD_SERIES, "Production (73 malicious, 1,419 not)", CROSS_LABEL["production"], (8, 2e5))]:
        ax.axvspan(1, 10, color="#f0efec", lw=0, zorder=0)
        for i, (lab, key, col, mk, ls, filled) in enumerate(series):
            m, fe = data[key]
            ax.plot(cs, cs * m + fe, color=col, ls=ls, lw=1.3, label=lab, marker=mk, markevery=(12 + 9 * i, 45), ms=4,
                    mfc=col if filled else "white", mec=col, mew=0.9)
        ax.axvline(cross, ymax=0.5, color=INK2, lw=0.7, ls=(0, (2, 2)))  # stops below the legend
        ax.text(cross * 1.08, 0.04, rf"$c^*{{\approx}}{cross}$", transform=ax.get_xaxis_transform(), color=INK2, fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, 300)

        def _plain(v, _):  # full-size tick labels; mathtext exponents would render below 9 pt
            return f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(FuncFormatter(_plain))
            axis.set_minor_formatter(NullFormatter())
        ax.set_ylim(*ylim)
        ax.set_xlabel("Miss cost $c$ (false escalations per miss)")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.grid(True, which="major", axis="y")
        ax.legend(frameon=False, loc="upper left", handlelength=2.4, labelspacing=0.12, fontsize=9, ncol=2, columnspacing=0.8)
    axes[0].set_ylabel(r"Expected cost $c\cdot$misses + FE")
    fig.savefig(out / "cost_vs_c_final.pdf")
    fig.savefig(out / "cost_vs_c_final.png", dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guide", type=Path, default=ROOT / "aggregates" / "guide")
    ap.add_argument("--production", type=Path, default=ROOT / "aggregates" / "production")
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    gfull, ggate, pfull, pgate = load(a.guide, a.production)
    g_cross = best_crossover(ggate, LOCAL_GUIDE, "Decisions + exemplar")
    p_cross = best_crossover(pgate, LOCAL_PROD, "Decisions + exemplar")
    assert round(g_cross) == CROSS_LABEL["GUIDE"], g_cross
    assert abs(p_cross - CROSS_LABEL["production"]) <= 0.5, p_cross  # 21.5 (RiskFit rho=0.8); MaxSim 21.4
    figure1(gfull, pfull, a.out)
    figure2(ggate, pgate, a.out)
    print("Figure 1, GUIDE:", {k: gfull[k] for _, k in GUIDE_BARS})
    print("Figure 1, production:", {k: pfull[k] for _, k in PROD_BARS})
    print(f"Figure 2: best-local crossover with Decisions + exemplar: GUIDE {g_cross:.1f}, production {p_cross:.1f}")
    print(f"wrote {a.out / 'default_all_final.pdf'}, {a.out / 'cost_vs_c_final.pdf'}, {a.out / 'cost_vs_c_final.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
