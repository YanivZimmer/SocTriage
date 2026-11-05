"""Release checks that need no training, no model download and no API call.

1. Production aggregates: every file is per tenant (A/B/C) or pooled, carries no fold-level, run-id or
   image columns, and its key counts equal the numbers printed in the paper's tables.
2. GUIDE snapshot (aggregates/guide): the paper's GUIDE numbers are present in paper_numbers.txt.
3. Figures 2 and 3 regenerate from the released CSVs.
4. Optional (--splits): scripts/analyze_guide.py on the released per-prompt predictions and outputs reproduces
   the snapshot's pooled gate counts for all twelve methods.

Usage: python tests/test_release.py [--splits data/guide_splits/splits]
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROD = ROOT / "aggregates" / "production"
GUIDE = ROOT / "aggregates" / "guide"
SCOPE_COLS = ("scope", "tenant")
ALLOWED_SCOPES = {"A", "B", "C", "all", "median over 9 tenant-folds", "tenant", ""}
FORBIDDEN_COLS = re.compile(r"^(fold\d*|fold|pipeline.*|.*image.*|.*digest.*|vertex.*|job.*|alert_id|id|prompt_id)$", re.I)


def rows(p: Path):
    return list(csv.DictReader(p.open(newline="")))


def check_production() -> None:
    files = sorted(PROD.glob("*.csv"))
    assert files, "no production aggregates"
    for f in files:
        rr = rows(f)
        cols = rr[0].keys() if rr else []
        bad = [c for c in cols if FORBIDDEN_COLS.match(c)]
        assert not bad, f"{f.name}: per-record or internal columns {bad}"
        for c in SCOPE_COLS:
            if c in cols:
                vals = {r[c] for r in rr}
                assert vals <= ALLOWED_SCOPES, f"{f.name}: unexpected {c} values {vals - ALLOWED_SCOPES}"
    # spot checks against the paper's tables
    dv = {r["method"]: r for r in rows(PROD / "default_verdicts.csv") if r["scope"] == "all"}
    assert (dv["RiskFit"]["caught"], dv["RiskFit"]["false_esc"]) == ("56", "64")
    gate = {r["method"]: (int(r["misses"]), int(r["false_esc"])) for r in rows(PROD / "gate80_test_quantile.csv") if r["scope"] == "all"}
    table5 = {"Decisions + exemplar": (0, 154), "RiskFit rho=0.8": (6, 25), "MaxSim": (7, 4), "Focal-Tversky": (7, 18),
              "RiskFit lambda=4": (8, 31), "RiskFit": (9, 28), "CE": (9, 8), "Qwen3-LoRA": (10, 5),
              "Gemini + exemplar": (11, 41), "LightGBM": (13, 6), "kNN (frozen)": (16, 2), "SetFit-LoRA": (16, 4),
              "Gemini zero-shot": (21, 181), "Decisions zero-shot": (1, 945)}
    assert all(gate[m] == v for m, v in table5.items()), "Table 5 counts"
    data = {r["tenant"]: r for r in rows(PROD / "data_summary.csv")}
    assert [data[t]["alerts"] for t in ("A", "B", "C", "all")] == ["412", "707", "373", "1492"]
    assert [data["all"][k] for k in ("false_positive", "tp_benign", "tp_malicious", "analyst_corrections", "malicious_corrections")] \
        == ["655", "764", "73", "528", "6"]
    ev = {(r["method"], r["condition"]): r["unreviewed_misses"] for r in rows(PROD / "evasion.csv") if r["gate"] == "test-quantile 80%"}
    assert [ev[("LightGBM", c)] for c in ("none", "P1", "P2", "P3", "P4", "R0")] == ["13", "13", "13", "23", "23", "13"]
    ws = {(r["method"], r["arm"], r["k_pct"]): r for r in rows(PROD / "weak_supervision.csv")}
    w = ws[("RiskFit", "weak (LLM verdicts + revealed labels)", "25")]
    assert (w["caught"], w["false_esc"]) == ("61", "556")
    print(f"[PASS] production aggregates: {len(files)} files, tenant-level or pooled only, table spot checks")


def check_guide_snapshot() -> None:
    txt = (GUIDE / "paper_numbers.txt").read_text()
    for s in ("431/79", "428/100*", "328/101", "350/113", "11/46", "24/109", "38/245", "0/502", "124.7", "c* 896.3",
              "c* 695.4", "caught 410 with 75", "caught 386 with 68", "0.917", "expected 16.5", "1077 of 1079"):
        assert s in txt, f"paper number {s!r} missing from paper_numbers.txt"
    print("[PASS] GUIDE snapshot contains the paper's GUIDE numbers")


def check_figures() -> None:
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_figures.py"), "--out", d], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
        for f in ("default_verdicts.pdf", "cost_vs_c.pdf", "cost_vs_c.png"):
            assert (Path(d) / f).stat().st_size > 1000, f
    print("[PASS] Figures 2 and 3 regenerate from released CSVs")


def check_analysis(splits: Path) -> None:
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_guide.py"), "--predictions", str(ROOT / "data" / "guide_predictions"),
                        "--splits", str(splits), "--out", d, "--n-boot", "50"], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        key = lambda rr: {(r["gate"], r["method"]): (r["misses"], r["false_esc"]) for r in rr}  # noqa: E731
        got, want = key(rows(Path(d) / "frontier_cost.csv")), key(rows(GUIDE / "frontier_cost.csv"))
        assert got == want and len(got) == 22, "gate counts differ from the released snapshot"
    print("[PASS] analyze_guide.py reproduces the snapshot's gate counts for all 12 methods")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=Path, help="data/guide_splits/splits (from scripts/build_guide_splits.py)")
    a = ap.parse_args()
    check_production()
    check_guide_snapshot()
    check_figures()
    if a.splits:
        check_analysis(a.splits)
    print("ALL RELEASE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
