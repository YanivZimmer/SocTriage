"""Release checks that need no training, no model download and no API call.

1. Production aggregates (aggregates/production/*.csv): every file is per tenant (A/B/C) or pooled, carries no
   fold-level, run-id or image columns, and its key counts equal the numbers printed in the final paper.
   Tenant x fold x seed count tables (aggregates/production/units/) carry only tenant A/B/C, fold 0-2, seed and
   counts or rates, never an alert, prompt or run identifier.
2. GUIDE snapshot (aggregates/guide): the final paper's GUIDE numbers are present in paper_numbers.txt.
3. Figures 1 and 2 regenerate from the released CSVs.
4. The accumulation-window recall hinge (riskfit/models/fastfit/modeling.py) has the gradient of the one-graph
   window hinge (synthetic tensors; needs torch, skipped without it).
5. Every released result file is mapped in README.md, and no file matches the generic leak patterns below.
6. Optional (--splits): scripts/analyze_guide.py on the released per-prompt predictions and outputs reproduces
   the snapshot's gate counts for all 18 GUIDE conditions.

Usage: python tests/test_release.py [--splits data/guide_splits/splits]
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROD = ROOT / "aggregates" / "production"
GUIDE = ROOT / "aggregates" / "guide"
SCOPE_COLS = ("scope", "tenant")
ALLOWED_SCOPES = {"A", "B", "C", "all", "median over 9 tenant-folds", "tenant", ""}
FORBIDDEN_COLS = re.compile(r"^(fold\d*|fold|pipeline.*|.*image.*|.*digest.*|vertex.*|job.*|alert_id|id|prompt_id|.*sha256.*|run.*|gcs.*|path)$", re.I)
UNIT_COLS_FORBIDDEN = re.compile(r"^(pipeline.*|.*image.*|.*digest.*|vertex.*|job.*|alert_id|id|prompt_id|.*sha256.*|run.*|gcs.*|path|source)$", re.I)


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
    units = sorted((PROD / "units").glob("*.csv"))
    for f in units:
        rr = rows(f)
        bad = [c for c in rr[0] if UNIT_COLS_FORBIDDEN.match(c)]
        assert not bad, f"units/{f.name}: identifier columns {bad}"
        assert {r["tenant"] for r in rr} <= {"A", "B", "C"} and {r["fold"] for r in rr} <= {"0", "1", "2"}, f.name
        assert all(re.fullmatch(r"\d+", r["seed"]) for r in rr), f.name
    # Figure 1 (right) and Table 1, full coverage
    dv = {r["method"]: (int(r["caught"]), int(r["false_esc"])) for r in rows(PROD / "default_verdicts.csv") if r["scope"] == "all"}
    assert dv["RiskFit"] == (56, 64) and dv["Decisions + exemplar"] == (71, 213) and dv["Tiny Decision"] == (45, 14)
    assert dv["CE-A100 (lambda=0)"] == (50, 28) and 10 * (73 - 56) + 64 == 234 and 10 * (73 - 71) + 213 == 233
    # Table 4 (production gate) and Table 1 gate costs
    gate = {r["method"]: (int(r["misses"]), int(r["false_esc"])) for r in rows(PROD / "gate80_test_quantile.csv") if r["scope"] == "all"}
    table4 = {"Decisions + exemplar": (0, 154), "RiskFit rho=0.8": (6, 25), "MaxSim": (7, 4), "Focal-Tversky": (7, 18),
              "RiskFit lambda=4": (8, 31), "RiskFit": (9, 28), "CE": (9, 8), "Tiny Decision": (10, 5),
              "Gemini + exemplar": (11, 41), "LightGBM": (13, 6), "kNN (frozen)": (16, 2), "SetFit-LoRA": (16, 4),
              "Gemini zero-shot": (21, 181), "Decisions zero-shot": (1, 945)}
    assert all(gate[m] == v for m, v in table4.items()), "Table 4 counts"
    assert [10 * gate[m][0] + gate[m][1] for m in ("RiskFit", "CE", "Tiny Decision", "Decisions + exemplar")] == [118, 98, 105, 154]
    data = {r["tenant"]: r for r in rows(PROD / "data_summary.csv")}
    assert [data[t]["alerts"] for t in ("A", "B", "C", "all")] == ["412", "707", "373", "1492"]
    assert [data[t]["tp_malicious"] for t in ("A", "B", "C")] == ["17", "39", "17"]
    assert [data["all"][k] for k in ("analyst_corrections", "malicious_corrections", "uncorrected", "uncorrected_equal_llm_export_verdict")] \
        == ["528", "6", "964", "943"]
    # Table 3 (recall hinge by sampling)
    arms = {(r["sampling"], r["loss"]): r for r in rows(PROD / "hinge_by_sampling_arms.csv")}
    t3 = {("balanced", "CE"): ("47.3", "27.0", "137"), ("natural", "CE"): ("11.7", "1.7", "227"),
          ("natural", "rho=0.99 lambda=8"): ("47.0", "28.7", "83")}
    for k, (c, fe, g) in t3.items():
        r = arms[k]
        assert (f"{float(r['caught']):.1f}", f"{float(r['fe']):.1f}", f"{float(r['cost_gate80']):.0f}") == (c, fe, g), k
    # seven seeds, recall-target sweep, beta note, 99.98%
    t = {r["metric"]: r for r in rows(PROD / "seeds_riskfit_vs_ce_7seeds_tests.csv")}
    assert (t["caught"]["riskfit_pooled_mean"], t["caught"]["ce_pooled_mean"], t["fe"]["pooled_diff"]) == ("50.000", "49.429", "9.000")
    sd = {r["method"]: r for r in rows(PROD / "seed_deployable_cost.csv")}
    assert all(int(sd[m]["max"]) < 155 for m in sd) and round(float(sd["RiskFit"]["mean"])) == 104
    mono = rows(PROD / "recall_target_sweep_monotonicity.csv")[0]
    assert (mono["spearman"], mono["mean_R_rho0.7"], mono["mean_R_rho0.95"]) == ("0.008", "0.718", "0.737")
    b = next(r for r in rows(PROD / "beta025_seed_change.csv") if r["scope"] == "all")
    assert b["fe_mean_sd"] == "-15.67 +- 13.32"
    mc = next(r for r in rows(PROD / "gate80_min_cost_probability.csv") if r["method"].startswith("any of the 10"))
    assert mc["prob_min_cost_ties_split"] == "0.9998"
    ev = {(r["method"], r["condition"]): r["unreviewed_misses"] for r in rows(PROD / "evasion.csv") if r["gate"] == "test-quantile 80%"}
    assert [ev[("LightGBM", c)] for c in ("none", "P1", "P2", "P3", "P4", "R0")] == ["13", "13", "13", "23", "23", "13"]
    ws = {(r["method"], r["arm"], r["k_pct"]): r for r in rows(PROD / "weak_supervision.csv")}
    w = ws[("RiskFit", "weak (LLM verdicts + revealed labels)", "25")]
    assert (w["caught"], w["false_esc"]) == ("61", "556")
    print(f"[PASS] production aggregates: {len(files)} tenant-level/pooled files and {len(units)} tenant x fold x seed "
          "count tables, column rules and final-paper spot checks")


def check_guide_snapshot() -> None:
    txt = (GUIDE / "paper_numbers.txt").read_text()
    for s in ("RiskFit 156 / CE 185; Tiny Decision 150", "Decisions + 20 exemplars 210", "RiskFit 329; Decisions + exemplar 740",
              "RiskFit 124.7 / CE 55.0", "12/30", "12/90", "59/437", "inf /   18.3", "Tiny Decision 0.6380, RiskFit 0.3020",
              "c=10 0.9760, c=20 0.8675", "0.842 / 0.602", "8 [3-15], RiskFit 11 [5-19]", "0.585 / 0.523",
              "Decisions + 20 exemplars 541, Jev + 20 exemplars 875, Tiny Decision 325, RiskFit 380, CE 451",
              "caught 399-452, false escalations 126-620", "1077 of 1079", "decisions 20ex (vs most similar exemplar) 0.888",
              "420 -> 90 (79%", "381 -> 213 (44%", "caught 410 with 75", "c* 896.3", "c* 695.4", "expected 16.5",
              "kNN misses range 19-21"):
        assert s in txt, f"paper number {s!r} missing from paper_numbers.txt"
    print("[PASS] GUIDE snapshot contains the final paper's GUIDE numbers")


def check_figures() -> None:
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_figures.py"), "--out", d], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
        for f in ("default_all_final.pdf", "cost_vs_c_final.pdf", "cost_vs_c_final.png"):
            assert (Path(d) / f).stat().st_size > 1000, f
    print("[PASS] Figures 1 and 2 regenerate from released CSVs")


def check_window_hinge() -> None:
    try:
        import torch
    except ImportError:
        print("[SKIP] window-hinge gradient check (torch not installed)")
        return
    sys.path.insert(0, str(ROOT))
    from riskfit.models.fastfit.modeling import FastFitTrainable

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(5, 3)
            self.config = types.SimpleNamespace(recall_constraint_rho=0.9, recall_hinge_window="accum",
                                                false_positive_penalty_beta=0.0, sim_factor=0.5, recall_constraint_lambda=2.0)
    for name in ("hinge_window_begin", "_hinge_stats", "_hinge_unit", "_window_hinge"):
        setattr(Stub, name, getattr(FastFitTrainable, name))
    torch.manual_seed(0)
    X = [torch.randn(4, 5) for _ in range(3)]
    Y = [torch.tensor([2, 0, 1, 0]), torch.tensor([0, 0, 1, 0]), torch.tensor([2, 2, 0, 1])]
    ref_m = Stub()
    p = torch.cat([torch.softmax(ref_m.lin(x), 1)[:, 2] for x in X])
    (2.0 * torch.relu(0.9 - p[torch.cat(Y) == 2].mean())).backward()
    m = Stub()
    m.load_state_dict(ref_m.state_dict())
    m.hinge_window_begin(3)
    for x, y in zip(X, Y):
        pr = torch.softmax(m.lin(x), 1)[:, 2]
        (m._window_hinge(pr, y == 2, torch.tensor(2.0)) / 3 + 0 * pr.sum()).backward()
    assert all(torch.allclose(a.grad, b.grad, atol=1e-6) for a, b in zip(ref_m.parameters(), m.parameters()))
    print("[PASS] accumulation-window hinge gradient equals the one-graph window hinge")


_P = ["gs" + ":" + "//", "/" + "Users/", "/private" + "/tmp", "/" + "home/", "docker" + r"\.pkg\.dev"]  # split so this file is grep-clean
LEAK = re.compile("|".join(_P) + r"|\b(?:sk-[A-Za-z0-9]{8,}|AIza[0-9A-Za-z_-]{20,}|gh[po]_[A-Za-z0-9]{20,})"
                  r"|\b(?:\d{1,3}\.){3}\d{1,3}\b")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def check_tree() -> None:
    readme = (ROOT / "README.md").read_text()
    released = [p for p in (ROOT / "aggregates").rglob("*") if p.is_file()] + list((ROOT / "data" / "guide_outputs").glob("*.csv"))
    missing = [str(p.relative_to(ROOT)) for p in released if p.name not in readme]
    assert not missing, f"result files not mapped in README.md: {missing}"
    hits = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or ".git" in p.parts or p.suffix in (".pdf", ".png", ".pyc") or p.resolve() == Path(__file__).resolve():
            continue
        s = p.read_text(errors="ignore")
        hits += [(str(p.relative_to(ROOT)), m.group()) for m in LEAK.finditer(s)]
        hits += [(str(p.relative_to(ROOT)), e) for e in EMAIL.findall(s) if e != "yaniv.zimmer@torq.io" and not e.endswith(".py")]
    assert not hits, f"leak-pattern hits: {hits[:10]}"
    print(f"[PASS] {len(released)} released result files mapped in README.md; no leak-pattern hits")


def check_analysis(splits: Path) -> None:
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_guide.py"), "--predictions", str(ROOT / "data" / "guide_predictions"),
                        "--splits", str(splits), "--out", d, "--n-boot", "50"], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        key = lambda rr: {r["method"]: (r["misses"], r["false_esc"]) for r in rr}  # noqa: E731
        got, want = key(rows(Path(d) / "frontier_cost.csv")), key(rows(GUIDE / "frontier_cost.csv"))
        assert got == want and len(got) == 18, "gate counts differ from the released snapshot"
    print("[PASS] analyze_guide.py reproduces the snapshot's gate counts for all 18 GUIDE conditions")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=Path, help="data/guide_splits/splits (from scripts/build_guide_splits.py)")
    a = ap.parse_args()
    check_production()
    check_guide_snapshot()
    check_figures()
    check_window_hinge()
    check_tree()
    if a.splits:
        check_analysis(a.splits)
    print("ALL RELEASE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
