#!/usr/bin/env python3
"""All GUIDE statistics in the final paper, from local-model predictions and the released per-prompt outputs.

Inputs (local files only, no network):
  --predictions    directory with riskfit/, ce/, lightgbm/, maxsim/, knn/, tiny_decision/ subdirectories of
                   org_<id>.jsonl (scripts/train_guide.py; the paper's files are in data/guide_predictions)
  --hosted         directory with decisions_guide_per_prompt.csv, jev_guide_per_prompt.csv, opus_guide_per_prompt.csv
                   (default: data/guide_outputs, the released outputs)
  --five-exemplar  Decisions and Jev with five retrieved exemplars (default: the released
                   data/guide_outputs/decisions_jev_5ex_guide_per_prompt.csv)
  --twenty-exemplar Decisions and Jev with twenty retrieved exemplars (default: the released
                   data/guide_outputs/decisions_jev_20ex_guide_per_prompt.csv)
  --open-llm       CyberSecQwen-4B zero-shot (default: the released
                   data/guide_outputs/cybersecqwen4b_zeroshot_guide_per_prompt.csv, from scripts/run_cybersecqwen.py)
  --open-llm-base  its base model Qwen3-4B-Instruct-2507, zero-shot, same prompt and scoring (default: the released
                   data/guide_outputs/qwen3_4b_base_zeroshot_guide_per_prompt.csv)
  --splits         the splits/ directory (prompt ids, labels and the exemplars' training labels)

The 18 conditions of the paper's Table 2: six local models (RiskFit, CE, LightGBM, MaxSim, kNN and the tiny local
decision model, Qwen3-0.6B-LoRA, named "Tiny Decision"); Decisions and Jev zero-shot and with 1, 5 and 20
exemplars; Opus zero-shot and with one exemplar; CyberSecQwen-4B and Qwen3-4B base zero-shot. Opus scored test
prompts only, so validation-tuned thresholds are n/a for it. The bootstrap resamples prompt positions within each
organization independently of the method list, so adding or removing a method does not change any other number.

Outputs (--out): full_coverage, matched_fpr, auc, gate80, gate80_ties, frontier_cost, bootstrap_gate80,
bootstrap_cost_crossovers, bootstrap_min_cost_probability, paired_tests, exemplar_following_test, neighbor_vote,
crossovers_reweighted (.csv), summary.json, and paper_numbers.txt (every GUIDE number quoted in the final paper,
grouped by the table, figure or paragraph that uses it). Only statistics that the final paper reports are written.

Score conventions: local models rank the gate by max class probability and use p(malicious) for ranking;
decision models gate on their own verdict confidence and use the predicate probability (q2); Opus gates on its
verbalized confidence and ranks by its p_malicious; the open LLMs gate on their largest label probability and rank
by p(malicious). The gate is the 80% test-quantile gate (top 80% of each organization's test prompts by confidence).
Exact Wilcoxon p-values with zero differences need scipy >= 1.16.

Usage: python scripts/analyze_guide.py --predictions data/guide_predictions --splits data/guide_splits/splits --out results/guide
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from riskfit import evaluation as E  # noqa: E402
from riskfit import guide as G  # noqa: E402

LABELS, MAL, ism = E.LABELS, E.MAL, E.ism
C2X = {"false_positive": LABELS[0], "benign": LABELS[1], "malicious": LABELS[2], "": "unresolved", None: "unresolved"}
LOCAL = {"RiskFit": "riskfit", "CE": "ce", "LightGBM": "lightgbm", "MaxSim": "maxsim", "kNN": "knn",
         "Tiny Decision": "tiny_decision"}  # Tiny Decision = Qwen3-0.6B-LoRA, 10 training epochs on GUIDE
NEWLOCAL = ["MaxSim", "kNN", "Tiny Decision"]
DZS, DEX, JZS, JEX, OZS, OEX = ("Decisions zero-shot", "Decisions + exemplar", "Jev zero-shot", "Jev + exemplar",
                                "Opus zero-shot", "Opus + exemplar")
D5, J5, D20, J20 = "Decisions + 5 exemplars", "Jev + 5 exemplars", "Decisions + 20 exemplars", "Jev + 20 exemplars"
CSQ, Q4B = "CyberSecQwen-4B zero-shot", "Qwen3-4B base zero-shot"
DECISION = [DZS, DEX, JZS, JEX]
OPUS = [OZS, OEX]
FIVE, TWENTY = [D5, J5], [D20, J20]
OPEN = [CSQ, Q4B]  # open LLMs run locally, zero-shot; compared like a hosted model
HOSTED = DECISION + OPUS + FIVE + TWENTY + OPEN
METHODS = list(LOCAL) + HOSTED
POOL15 = [m for m in METHODS if m not in TWENTY + [Q4B]]  # the paper's "without the twenty-exemplar conditions"
HAS_VAL = [m for m in METHODS if m not in OPUS]
NA_VAL = "n/a: no validation scores (test prompts only)"
CS = (5, 10, 20, 50)
B, SEED = 2000, 42
GATE = "(a) per-org test top 80% by confidence"
PROD_MAL, PROD_NONMAL = 73, 1419  # production class ratio for prevalence reweighting


def fnum(x):
    return None if x in (None, "", "None") else float(x)


def parse_args():
    out = ROOT / "data" / "guide_outputs"
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--hosted", type=Path, default=out)
    ap.add_argument("--five-exemplar", type=Path, default=out / "decisions_jev_5ex_guide_per_prompt.csv")
    ap.add_argument("--twenty-exemplar", type=Path, default=out / "decisions_jev_20ex_guide_per_prompt.csv")
    ap.add_argument("--open-llm", type=Path, default=out / "cybersecqwen4b_zeroshot_guide_per_prompt.csv")
    ap.add_argument("--open-llm-base", type=Path, default=out / "qwen3_4b_base_zeroshot_guide_per_prompt.csv")
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--orgs", help="comma-separated subset (smoke runs); default: all 12")
    ap.add_argument("--n-boot", type=int, default=B)
    return ap.parse_args()


# ----------------------------------------------------------------------------- load
def decision_rec(r, pfx):
    """One decision-model row (unresolved -> confidence 0, scores 0)."""
    ok = r[f"{pfx}_verdict"] not in ("", None)
    return {"id": r["prompt_id"], "y": C2X[r["label"]], "pred": C2X[r[f"{pfx}_verdict"]] if ok else "unresolved",
            "probs": [fnum(r[f"{pfx}_p_fp"]) or 0.0, fnum(r[f"{pfx}_p_benign"]) or 0.0,
                      fnum(r[f"{pfx}_p_mal_q1"]) or 0.0] if ok else [0.0] * 3,
            "conf": fnum(r[f"{pfx}_confidence"]) if ok else 0.0, "q2": fnum(r[f"{pfx}_p_mal_q2"]) if ok else 0.0}


def load(a):
    all_orgs = [r["org_id"] for r in G.iter_orgs(ROOT / "configs" / "guide_orgs.csv")]
    orgs = a.orgs.split(",") if a.orgs else all_orgs
    splits = {o: G.load_split(a.splits, o) for o in orgs}
    truth = {(o, s): {(r["alert_id"], r["label"]) for r in splits[o][s]} for o in orgs for s in ("test", "val")}
    data = {}
    for m, d in LOCAL.items():
        for o in orgs:
            te, va = [], []
            for line in (a.predictions / d / f"org_{o}.jsonl").open():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["split"] not in ("test", "val"):
                    continue
                p = r["class_probs"]
                probs = [float(p.get(x, 0.0)) for x in LABELS]
                pred = r["predicted_label"]
                conf = float(p[pred])
                sc = r.get("confidence")
                if sc not in (None, "None") and abs(float(sc) - conf) <= 1e-5:
                    conf = float(sc)  # unrounded max probability
                rec = {"id": r["alert_id"], "y": r["expected_label"], "pred": pred, "probs": probs, "conf": conf, "q2": probs[2]}
                (te if r["split"] == "test" else va).append(rec)
            data[(m, o)] = {"test": te, "val": va}
    for fname, pairs in (("decisions_guide_per_prompt.csv", ((DZS, "zs"), (DEX, "ex"))),
                         ("jev_guide_per_prompt.csv", ((JZS, "zs"), (JEX, "ex")))):
        for r in csv.DictReader((a.hosted / fname).open(newline="")):
            if r["org"] not in orgs:
                continue
            for m, pfx in pairs:
                data.setdefault((m, r["org"]), {"test": [], "val": []})[r["split"]].append(decision_rec(r, pfx))
    opus = {}
    for r in csv.DictReader((a.hosted / "opus_guide_per_prompt.csv").open(newline="")):
        opus[(r["prompt_id"], r["condition"])] = r
    for m, cond in ((OZS, "zs"), (OEX, "ex")):
        for o in orgs:
            te = []
            for i, y in sorted(truth[(o, "test")]):
                r = opus.get((i, cond))
                ok = r is not None and r["verdict"] not in ("", None)
                pm = float(r["p_malicious"]) if ok else 0.0
                te.append({"id": i, "y": y, "pred": C2X[r["verdict"]] if ok else "unresolved", "probs": [0.0, 0.0, pm],
                           "conf": float(r["confidence"]) if ok else 0.0, "q2": pm})
            data[(m, o)] = {"test": te, "val": []}
    multi = {}  # (k, split, prompt_id) -> row of the 5- or 20-exemplar file
    for path, k, pairs in ((a.five_exemplar, 5, ((D5, "dec5"), (J5, "jev5"))),
                           (a.twenty_exemplar, 20, ((D20, "dec20"), (J20, "jev20")))):
        for r in csv.DictReader(path.open(newline="")):
            if r["org"] not in orgs:
                continue
            multi[(k, r["split"], r["prompt_id"])] = r
            for m, pfx in pairs:
                data.setdefault((m, r["org"]), {"test": [], "val": []})[r["split"]].append(decision_rec(r, pfx))
    for m, path in ((CSQ, a.open_llm), (Q4B, a.open_llm_base)):  # softmax over the three labels' summed log-probabilities
        for r in csv.DictReader(path.open(newline="")):
            if r["org"] not in orgs:
                continue
            ps = [float(r["p_fp"]), float(r["p_benign"]), float(r["p_malicious"])]
            assert abs(sum(ps) - 1) < 1e-6 and float(r["confidence"]) == max(ps), r["id"]
            rec = {"id": r["id"], "y": C2X[r["y"]], "pred": C2X[r["verdict"]], "probs": ps,
                   "conf": float(r["confidence"]), "q2": ps[2]}
            data.setdefault((m, r["org"]), {"test": [], "val": []})[r["split"]].append(rec)
    for o in orgs:  # integrity: every method has exactly the split's prompts and labels
        for s in ("test", "val"):
            for m in METHODS:
                if s == "val" and m in OPUS:
                    continue
                got = {(r["id"], r["y"]) for r in data[(m, o)][s]}
                assert got == truth[(o, s)], f"prompt id/label mismatch: {m} org {o} {s}"
    for m in METHODS:  # canonical order = split-file order (the bootstrap resamples positions in this order)
        for o in orgs:
            order = {r["alert_id"]: k for k, r in enumerate(splits[o]["test"])}
            data[(m, o)]["test"].sort(key=lambda r: order[r["id"]])
    return orgs, splits, data, multi


def main() -> int:
    a = parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    orgs, splits, DATA, MULTI = load(a)
    nb = a.n_boot

    def write(name, rows):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with (a.out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if r.get(k) is None else (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k])) for k in keys})

    S = {"orgs": orgs, "n_test_prompts": sum(len(DATA[("RiskFit", o)]["test"]) for o in orgs),
         "unresolved": {m: sum(r["pred"] == "unresolved" for o in orgs for s in ("test", "val") for r in DATA[(m, o)][s]) for m in METHODS}}

    # --------------------------------------------------------------------- full coverage (Figure 1 left; Table 1)
    fc_rows, FC = [], {}
    for m in METHODS:
        per = {o: E.full_coverage(DATA[(m, o)]["test"]) for o in orgs}
        FC[m] = per
        fc_rows += [{"method": m, "scope": f"org {o}", **per[o]} for o in orgs]
        pooled = E.full_coverage([r for o in orgs for r in DATA[(m, o)]["test"]])
        fc_rows.append({"method": m, "scope": "pooled (per prompt)", **pooled})
        S.setdefault("full_coverage", {})[m] = pooled
    write("full_coverage.csv", fc_rows)

    # --------------------------------------------------------------------- threshold at RiskFit's FPR + AUC
    def score(m, r):
        return r["q2"] if m in HOSTED else r["probs"][2]

    def sname_of(m):
        return "p_mal" if m in LOCAL else ("p_malicious" if m in OPUS + OPEN else "q2")

    mf_rows, auc_rows, MF = [], [], collections.defaultdict(dict)
    T = "RiskFit default FPR"
    for o in orgs:
        for m in METHODS:
            te, va = DATA[(m, o)]["test"], DATA[(m, o)]["val"]
            y = [ism(r["y"]) for r in te]
            s = [score(m, r) for r in te]
            auc_rows.append({"method": m, "score": sname_of(m), "org": o, "roc_auc": float(roc_auc_score(y, s)),
                             "pr_auc": float(average_precision_score(y, s)), "n": len(te), "n_mal": sum(y),
                             "n_distinct_scores": len(set(s)), "prevalence": sum(y) / len(y)})
            if m in OPUS:
                mf_rows.append({"method": m, "score": sname_of(m), "org": o, "target": T, "selection": "validation-tuned", "note": NA_VAL})
                continue
            thr = E.np_threshold([score(m, r) for r in va if not ism(r["y"])], FC["RiskFit"][o]["fpr"])
            row = {"method": m, "score": sname_of(m), "org": o, "target": T, "target_fpr": FC["RiskFit"][o]["fpr"],
                   "selection": "validation-tuned", **E.at_threshold(te, lambda r: score(m, r), thr)}
            mf_rows.append(row)
            MF[m][o] = row
    for m, per in list(MF.items()):
        tp = sum(r["tp"] for r in per.values()); npos = sum(r["n_mal"] for r in per.values())
        fp = sum(r["fp"] for r in per.values()); nneg = sum(r["n_neg"] for r in per.values())
        mf_rows.append({"method": m, "score": sname_of(m), "org": "pooled", "target": T, "selection": "validation-tuned",
                        "tp": tp, "n_mal": npos, "fp": fp, "n_neg": nneg, "recall": tp / npos, "fpr": fp / nneg,
                        "cost_c10": 10 * (npos - tp) + fp})
    write("matched_fpr.csv", mf_rows)
    for m in METHODS:
        rr = [r for r in auc_rows if r["method"] == m and r["org"] != "mean over orgs"]
        for k in ("roc_auc", "pr_auc"):
            mu, sd, n = E.mean_sd([r[k] for r in rr])
            auc_rows.append({"method": m, "score": sname_of(m), "org": "mean over orgs", "metric": k, "mean": mu, "sd": sd, "n_orgs": n})
    write("auc.csv", auc_rows)

    # --------------------------------------------------------------------- 80% test-quantile gate (Tables 1-2)
    g_rows, Gt = [], {}
    for m in METHODS:
        per = {o: E.gate(DATA[(m, o)]["test"], "a", DATA[(m, o)]["val"]) for o in orgs}
        Gt[m] = per
        g_rows += [{"gate": GATE, "method": m, "scope": f"org {o}", **per[o]} for o in orgs]
        tot = {k: sum(per[o][k] for o in orgs) for k in ("kept", "n", "n_mal", "n_nonmal", "misses", "false_esc", "mal_routed")}
        tot.update({"coverage": tot["kept"] / tot["n"], "system_recall": 1 - tot["misses"] / tot["n_mal"],
                    "fe_per_1k_nonmal": 1000 * tot["false_esc"] / tot["n_nonmal"]})
        Gt[m]["pooled"] = tot
        g_rows.append({"gate": GATE, "method": m, "scope": "pooled", **tot})
    write("gate80.csv", g_rows)

    t_rows = []
    for m in METHODS:
        acc = collections.Counter()
        for o in orgs:
            row = E.gate_ties(DATA[(m, o)]["test"])
            t_rows.append({"method": m, "scope": f"org {o}", **row})
            for k in ("n", "kk", "n_above_tau", "n_tied_at_tau", "n_tied_kept", "n_tied_routed", "misses_point_idbytes", "misses_min",
                      "misses_max", "misses_expected_random_ties", "fe_point_idbytes", "fe_min", "fe_max", "fe_expected_random_ties"):
                acc[k] += row[k]
        t_rows.append({"method": m, "scope": "pooled", **{k: (float(v) if "expected" in k else v) for k, v in acc.items()}})
    write("gate80_ties.csv", t_rows)

    pts = {m: (Gt[m]["pooled"]["misses"], Gt[m]["pooled"]["false_esc"]) for m in METHODS}
    fr_rows = []
    for m in METHODS:
        dom = E.dominated_by(pts, m)
        row = {"gate": GATE, "method": m, "misses": pts[m][0], "false_esc": pts[m][1],
               "coverage": Gt[m]["pooled"]["coverage"], "on_frontier": not dom, "dominated_by": "; ".join(dom)}
        row.update({f"cost_c{c}": c * pts[m][0] + pts[m][1] for c in CS})
        fr_rows.append(row)
    for c in CS:
        S.setdefault("cheapest_point", {})[f"c{c}"] = min(METHODS, key=lambda m: c * pts[m][0] + pts[m][1])
    write("frontier_cost.csv", fr_rows)

    # --------------------------------------------------------------------- bootstrap (2,000 replicates, seed 42)
    MISS, FE = E.bootstrap_gate_counts(DATA, orgs, METHODS, set(), n_boot=nb, seed=SEED, ref_method="RiskFit")
    tot = {m: (MISS[("a", m)].sum(1), FE[("a", m)].sum(1)) for m in METHODS}
    bg_rows = []
    for m in METHODS:
        ml, mh = E.pct95(tot[m][0]); fl, fh = E.pct95(tot[m][1])
        bg_rows.append({"gate": GATE, "method": m, "point_misses": pts[m][0], "misses_ci95_low": ml, "misses_ci95_high": mh,
                        "point_false_esc": pts[m][1], "false_esc_ci95_low": fl, "false_esc_ci95_high": fh})
    write("bootstrap_gate80.csv", bg_rows)

    cx_rows, mc_rows = [], []
    for dm in HOSTED:
        Md, Fd = tot[dm]
        for lm in LOCAL:
            kd, cpt = E.region_local_cheaper(*pts[lm], *pts[dm])
            cx_rows.append({"gate": GATE, "pair": f"{lm} vs {dm}", "point_misses_local": pts[lm][0], "point_fe_local": pts[lm][1],
                            "point_misses_hosted": pts[dm][0], "point_fe_hosted": pts[dm][1],
                            "point_region_local_cheaper": kd, "point_c_star": cpt, **E.crossover_bootstrap(*tot[lm], Md, Fd)})
        Ml = np.stack([tot[lm][0] for lm in LOCAL], 1); Fl = np.stack([tot[lm][1] for lm in LOCAL], 1)
        pcs = [E.region_local_cheaper(*pts[lm], *pts[dm]) for lm in LOCAL]
        pt_env = max([c for k, c in pcs if k == "below"], default=None)
        arg = [lm for lm, (k, c) in zip(LOCAL, pcs) if k == "below" and c == pt_env]
        dom = [lm for lm, (k, c) in zip(LOCAL, pcs) if k == "always"]
        if dom:
            pt_env, arg = math.inf, [f"{x} (dominates: cheaper at every c)" for x in dom]
        cx_rows.append({"gate": GATE, "pair": f"best local (envelope of the six local models) vs {dm}", "point_c_star": pt_env,
                        "point_region_local_cheaper": "always" if dom else ("below" if pt_env else "never"),
                        "point_note": f"attained by {', '.join(arg)}" if arg else "", **E.envelope_bootstrap(Ml, Fl, Md, Fd, nb)})
    for pool_name, pool in ((f"{len(POOL15)} methods (all but the 20-exemplar conditions and Qwen3-4B base)", POOL15),
                            (f"all {len(METHODS)} methods", METHODS)):
        for c in CS:
            for m, s in zip(pool, E.min_cost_probability(MISS, FE, "a", pool, c)):
                mc_rows.append({"gate": GATE, "candidate_pool": pool_name, "c": c, "method": m,
                                "point_cost": c * pts[m][0] + pts[m][1], "prob_min_cost_ties_split": float(s)})
    write("bootstrap_cost_crossovers.csv", cx_rows)
    write("bootstrap_min_cost_probability.csv", mc_rows)

    for lm in ("RiskFit", "CE"):  # paper: RiskFit / CE dominate Opus + exemplar (fewer or equal misses and FE)
        dom = (tot[lm][0] <= tot[OEX][0]) & (tot[lm][1] <= tot[OEX][1])
        S.setdefault("share_local_weakly_dominates", {})[f"{lm} vs {OEX}"] = float(dom.mean())

    # --------------------------------------------------------------------- paired tests over organizations (Holm)
    def org_metric(m, metric):
        out = []
        for o in orgs:
            if metric == "matched_fpr_recall":
                out.append(MF[m][o]["recall"])
            elif metric == "gate_miss_rate":
                g = Gt[m][o]; out.append(g["misses"] / g["n_mal"])
            elif metric == "gate_fe_rate":
                g = Gt[m][o]; out.append(g["false_esc"] / g["n_nonmal"])
            elif metric == "matched_val_fpr_recall":
                out.append(matched_val_fpr_recall(m, o))
            elif metric == "mal_recall":
                out.append(FC[m][o]["recall"])
            elif metric == "kappa":
                out.append(FC[m][o]["kappa"])
        return out

    def matched_val_fpr_recall(m, o):
        """Pre-registered local-model rule: RiskFit at its argmax operating point; for CE / LightGBM the smallest
        threshold t (flag iff p_mal >= t) whose validation FPR <= RiskFit's validation FPR, applied to test."""
        if m == "RiskFit":
            return FC["RiskFit"][o]["recall"]
        rv = DATA[("RiskFit", o)]["val"]
        target = sum(ism(r["pred"]) for r in rv if not ism(r["y"])) / sum(not ism(r["y"]) for r in rv)
        va, te = DATA[(m, o)]["val"], DATA[(m, o)]["test"]
        neg = [r["probs"][2] for r in va if not ism(r["y"])]
        t = next(t for t in sorted(set(r["probs"][2] for r in va)) + [math.inf]
                 if sum(x >= t for x in neg) / len(neg) <= target)
        pos = [r for r in te if ism(r["y"])]
        return sum(r["probs"][2] >= t for r in pos) / len(pos)

    def family(set_name, fam, comps, metrics):
        rows = []
        for hm, ref in comps:
            for metric in metrics:
                x = org_metric(hm, metric); y = org_metric(ref, metric)
                r = E.paired(x, y)
                rows.append({"set": set_name, "family": fam, "comparison": f"{hm} - {ref}", "metric": metric,
                             **{k: v for k, v in r.items() if not isinstance(v, list)},
                             "mean_ci95_low": r["mean_ci95"][0], "mean_ci95_high": r["mean_ci95"][1],
                             "median_ci95_low": r["median_ci95"][0], "median_ci95_high": r["median_ci95"][1],
                             "a_mean": float(np.mean(x)), "ref_mean": float(np.mean(y)),
                             "per_org_a": " ".join(f"{v:.3f}" for v in x), "per_org_ref": " ".join(f"{v:.3f}" for v in y)})
        for r, p_, ps in zip(rows, E.holm([r["wilcoxon_p"] if not math.isnan(r["wilcoxon_p"]) else 1.0 for r in rows]),
                             E.holm([r["sign_p"] if not math.isnan(r["sign_p"]) else 1.0 for r in rows])):
            r["wilcoxon_p_holm"], r["sign_p_holm"] = p_, ps
        return rows

    pt = []
    if len(orgs) >= 2:
        pt += family("local models", "RQ1: recall, matched-FPR recall, kappa (Holm over 6)",
                     [("RiskFit", "CE"), ("RiskFit", "LightGBM")], ("mal_recall", "matched_val_fpr_recall", "kappa"))
        M3 = ("matched_fpr_recall", "gate_miss_rate", "gate_fe_rate")
        for set_name, comps in (("Decisions + exemplar vs local", [(DEX, "RiskFit"), (DEX, "CE")]),
                                ("Jev + exemplar vs local", [(JEX, "RiskFit"), (JEX, "CE")]),
                                ("local baselines vs RiskFit", [(m, "RiskFit") for m in NEWLOCAL])):
            pt += family(set_name, "gate (a), hosted q2 (Holm over comparisons x 3 metrics)", comps, M3)
        write("paired_tests.csv", pt)

    # --------------------------------------------------------------------- exemplar following (RQ2 'Exemplars')
    rows_h = {name: {(r["split"], r["prompt_id"]): r for r in csv.DictReader((a.hosted / fname).open(newline="")) if r["org"] in orgs}
              for fname, name in (("decisions_guide_per_prompt.csv", "decisions"), ("jev_guide_per_prompt.csv", "jev"))}
    ppd = rows_h["decisions"]
    ppo = {}
    for m, pfx in ((OZS, "zs"), (OEX, "ex")):
        for o in orgs:
            for r in DATA[(m, o)]["test"]:
                v = {LABELS[0]: "false_positive", LABELS[1]: "benign", LABELS[2]: "malicious"}.get(r["pred"], "")
                ppo.setdefault(("test", r["id"]), {})[f"{pfx}_verdict"] = v
    lab = {r["alert_id"]: C2X[{"False Positive": "false_positive", "True Positive - Benign": "benign",
                               "True Positive - Malicious": "malicious"}[r["label"]]] for o in orgs for r in splits[o]["train"]}
    fr = []
    for o in orgs + ["pooled"]:
        keys = [k for k in ppd if k[0] == "test" and (o == "pooled" or ppd[k]["org"] == o)]
        exl = {k: lab[ppd[k]["exemplar_prompt_id"]] for k in keys}
        tl = {k: C2X[ppd[k]["label"]] for k in keys}

        def add(name, cond, verdicts):
            ok = [k for k in keys if verdicts.get(k)]
            fr.append({"org": o, "model": name, "condition": cond, "n_resolved": len(ok),
                       "exemplar_label_correct_rate": sum(exl[k] == tl[k] for k in keys) / len(keys),
                       "exemplar_following_rate_3class": sum(C2X[verdicts[k]] == exl[k] for k in ok) / len(ok) if ok else None})

        for name, pp in (("decisions", rows_h["decisions"]), ("jev", rows_h["jev"]), ("opus", ppo)):
            for pfx in ("zs", "ex"):
                add(name, pfx, {k: pp[k][f"{pfx}_verdict"] for k in keys})
        for kx, pairs in ((5, (("decisions", "dec5"), ("jev", "jev5"))), (20, (("decisions", "dec20"), ("jev", "jev20")))):
            # k exemplars: agreement with the most similar one (= the one-exemplar condition's exemplar)
            assert all(MULTI[(kx, *k)]["ex1_prompt_id"] == ppd[k]["exemplar_prompt_id"] for k in keys), "top-1 exemplar differs"
            for name, pfx in pairs:
                add(name, f"{kx}ex (vs most similar exemplar)", {k: MULTI[(kx, *k)][f"{pfx}_verdict"] for k in keys})
    write("exemplar_following_test.csv", fr)

    # model-free vote: the most similar training prompt's label (exemplar ids from the one-exemplar file)
    nv = []
    for o in orgs + ["pooled"]:
        keys = [k for k in ppd if k[0] == "test" and (o == "pooled" or ppd[k]["org"] == o)]
        tl = {k: C2X[ppd[k]["label"]] for k in keys}
        flag = {k: ism(lab[ppd[k]["exemplar_prompt_id"]]) for k in keys}
        cau = sum(flag[k] for k in keys if ism(tl[k])); nm = sum(ism(tl[k]) for k in keys)
        fm = sum(flag[k] for k in keys if not ism(tl[k])); nn = len(keys) - nm
        nv.append({"org": o, "rule": "top-1 neighbor label", "caught": cau, "n_mal": nm, "false_mal": fm, "n_nonmal": nn,
                   **{f"cost_c{c}": c * (nm - cau) + fm for c in CS}})
    write("neighbor_vote.csv", nv)

    # --------------------------------------------------------------------- crossovers reweighted to production prevalence
    nm_g = sum(ism(r["y"]) for o in orgs for r in DATA[("RiskFit", o)]["test"])
    nn_g = sum(not ism(r["y"]) for o in orgs for r in DATA[("RiskFit", o)]["test"])
    w_prod = E.reweight_factor(PROD_MAL, PROD_NONMAL, nm_g, nn_g)
    rng = np.random.default_rng(SEED)
    sizes = {o: len(DATA[("RiskFit", o)]["test"]) for o in orgs}
    RW = ["RiskFit", "CE", DEX]

    def pooled_w(m, w, idx=None):
        mi = fe = 0.0
        for o in orgs:
            te = DATA[(m, o)]["test"]
            if idx is not None:
                te = [te[k] for k in idx[o]]
            x, y = E.weighted_gate(te, w); mi += x; fe += y
        return mi, fe

    rw_rows = []
    for w, tag in ((1.0, None), (w_prod, "GUIDE reweighted to production class ratio (73:1419)")):
        # the unweighted pass is not reported; it is run so that the reweighted pass draws the same replicates as before
        pw = {m: pooled_w(m, w) for m in RW}
        boots = collections.defaultdict(list)
        for _ in range(nb):
            idx = {o: rng.integers(0, sizes[o], sizes[o]) for o in orgs}
            if tag is None:
                continue
            pb = {m: pooled_w(m, w, idx) for m in RW}
            for L in ("RiskFit", "CE"):
                boots[L].append(E.cstar(*pb[L], *pb[DEX]))
        if tag is None:
            continue
        for L in ("RiskFit", "CE"):
            bs = np.array(boots[L])
            with np.errstate(invalid="ignore"):
                lo, hi = (float(np.quantile(np.where(np.isfinite(bs), bs, np.inf), q)) for q in (0.025, 0.975))
            row = {"analysis": tag, "local": L, "hosted": DEX, "weight_nonmal": w, "local_misses": pw[L][0], "local_fe": pw[L][1],
                   "hosted_misses": pw[DEX][0], "hosted_fe": pw[DEX][1], "c_star": E.cstar(*pw[L], *pw[DEX]),
                   "boot_lo": lo, "boot_hi": hi, "share_infinite": float(np.mean(~np.isfinite(bs)))}
            if row["share_infinite"] > 0.025:
                row["boot_hi"] = math.inf
            if row["share_infinite"] > 0.975:
                row["boot_lo"] = math.inf
            rw_rows.append(row)
    write("crossovers_reweighted.csv", rw_rows)
    S["reweight_factor_nonmal"] = w_prod
    (a.out / "summary.json").write_text(json.dumps(S, indent=1, default=str))

    # --------------------------------------------------------------------- the final paper's GUIDE numbers
    def g(m):
        return f"{pts[m][0]}/{pts[m][1]}"

    def gcost(m, c=10):
        return c * pts[m][0] + pts[m][1]

    def auc_ms(m):
        r = next(r for r in auc_rows if r["method"] == m and r.get("metric") == "roc_auc")
        return r["mean"], r["sd"]

    def cxr(pair):
        return next(x for x in cx_rows if x["pair"] == pair)

    def cx(pair):
        r = cxr(pair)
        return math.inf if r["point_region_local_cheaper"] == "always" else r["point_c_star"]

    def fmt_c(v):
        return "inf" if v is None or v == math.inf else f"{v:.1f}"

    def bci(m):
        r = next(x for x in bg_rows if x["method"] == m)
        return f"{r['misses_ci95_low']:.0f}-{r['misses_ci95_high']:.0f}"

    def pool_share(pool, c, who=None):
        rows = [r for r in mc_rows if r["candidate_pool"] == pool and r["c"] == c]
        return sum(r["prob_min_cost_ties_split"] for r in rows if (r["method"] in LOCAL if who is None else r["method"] == who))

    def fc_cost(m):
        p = S["full_coverage"][m]; return 10 * (p["n_mal"] - p["caught"]) + p["false_mal"]

    def tuned_cost(m):
        r = next(x for x in mf_rows if x["method"] == m and x["org"] == "pooled"); return r["cost_c10"]

    P15, P18 = mc_rows[0]["candidate_pool"], mc_rows[-1]["candidate_pool"]
    lines = [f"GUIDE organizations: {len(orgs)}; test prompts {S['n_test_prompts']} ({nm_g} malicious, {nn_g} non-malicious)",
             "Every line names the place in the final paper (AICS 2027) that quotes the number. Gate = 80% test-quantile gate.", ""]

    lines += ["Table 1 (GUIDE column), cost 10*misses+FE at c=10:",
              f"  80% gate: RiskFit {gcost('RiskFit')} / CE {gcost('CE')}; Tiny Decision {gcost('Tiny Decision')}; "
              f"{DEX} {gcost(DEX)}; {D20} {gcost(D20)}",
              f"  full coverage: RiskFit {fc_cost('RiskFit')}; {DEX} {fc_cost(DEX)}",
              f"  crossover c* vs {DEX}: RiskFit {fmt_c(cx(f'RiskFit vs {DEX}'))} / CE {fmt_c(cx(f'CE vs {DEX}'))}", ""]

    lines += ["Table 2 (tab:guidegate): 80% gate misses/FE, c* vs RiskFit and CE (inf = cheaper at every c), ROC-AUC mean +- sd over orgs"]
    for m in METHODS:
        mu, sd = auc_ms(m)
        cs_ = ("--", "--") if m in LOCAL else (fmt_c(cx(f"RiskFit vs {m}")), fmt_c(cx(f"CE vs {m}")))
        auc_s = f"{mu:.3f} +- {sd:.3f}" if sd is not None else f"{mu:.3f}"
        lines.append(f"  {m:28s} {g(m):>9s}  cost c=10 {gcost(m):5d}  c* {cs_[0]:>6s} / {cs_[1]:>6s}   AUC {auc_s}")
    kr = next(r for r in t_rows if r["method"] == "kNN" and r["scope"] == "pooled")
    lines.append(f"  caption: kNN misses range {kr['misses_min']}-{kr['misses_max']} with tie order at the cutoff")
    lines.append("")

    lines += ["Results, GUIDE 'Confidence gating':"]
    for c in (5, 10, 20):
        lines.append(f"  cheapest point estimate at c={c}: {min(METHODS, key=lambda m: gcost(m, c))}")
    lines.append(f"  pool '{P15}', c=10, share cheapest: Tiny Decision {pool_share(P15, 10, 'Tiny Decision'):.4f}, "
                 f"RiskFit {pool_share(P15, 10, 'RiskFit'):.4f}, CE {pool_share(P15, 10, 'CE'):.4f}")
    lines.append(f"  pool '{P18}': a local model cheapest at c=10 {pool_share(P18, 10):.4f}, c=20 {pool_share(P18, 20):.4f}")
    lines.append(f"  RiskFit / CE weakly dominate {OEX} in {S['share_local_weakly_dominates']['RiskFit vs ' + OEX]:.3f} / "
                 f"{S['share_local_weakly_dominates']['CE vs ' + OEX]:.3f} of replicates")
    lines.append(f"  {DEX} misses {pts[DEX][0]} [{bci(DEX)}], RiskFit {pts['RiskFit'][0]} [{bci('RiskFit')}]; "
                 f"FE ratio {pts[DEX][1] / max(pts['RiskFit'][1], 1):.1f}x")
    r = cxr(f"RiskFit vs {DEX}")
    lines.append(f"  RiskFit vs {DEX}: c* {fmt_c(cx(f'RiskFit vs {DEX}'))}, 95% interval {r['boot_full_p2.5 (inf=local always cheaper)']:.0f} "
                 f"to {fmt_c(r['boot_full_p97.5'])}")
    lines.append(f"  {D20}: {g(D20)}, cost {gcost(D20)}; RiskFit / Tiny Decision cheaper at every c in "
                 f"{cxr(f'RiskFit vs {D20}')['boot_share_local_cheaper_at_all_c']:.3f} / "
                 f"{cxr(f'Tiny Decision vs {D20}')['boot_share_local_cheaper_at_all_c']:.3f} of replicates; "
                 f"CE c* {fmt_c(cx(f'CE vs {D20}'))}")
    lines.append(f"  {CSQ}: RiskFit c* {fmt_c(cx(f'RiskFit vs {CSQ}'))}; {Q4B} ROC-AUC {auc_ms(Q4B)[0]:.3f}")
    lines.append("  best-local crossover with one-exemplar conditions: " + ", ".join(
        f"{h} {fmt_c(cx(f'best local (envelope of the six local models) vs {h}'))}" for h in (DEX, JEX, OEX)))
    lines.append("")

    lines += ["Figure 1 (left) and Results, GUIDE 'Default and tuned operating points': default verdicts, caught / false escalations"]
    for m in METHODS:
        p = S["full_coverage"][m]
        lines.append(f"  {m:28s} {p['caught']}/{p['false_mal']}  cost c=10 {fc_cost(m)}")
    hz = [S["full_coverage"][m] for m in HOSTED if m not in OPEN]
    lines.append(f"  hosted models: caught {min(p['caught'] for p in hz)}-{max(p['caught'] for p in hz)}, "
                 f"false escalations {min(p['false_mal'] for p in hz)}-{max(p['false_mal'] for p in hz)}")
    nmal = sum(ism(r["pred"]) for o in orgs for r in DATA[(CSQ, o)]["test"])
    lines.append(f"  CyberSecQwen-4B labeled {nmal} of {S['n_test_prompts']} test prompts malicious")
    lines.append("  cost at c=10 with a validation-tuned threshold at RiskFit's FPR: " + ", ".join(
        f"{m} {tuned_cost(m)}" for m in (D5, J5, D20, J20, "Tiny Decision", "RiskFit", "CE")))
    loc_min = min(auc_ms(m)[0] for m in LOCAL)
    lines.append(f"  lowest local ROC-AUC {loc_min:.3f}; hosted ROC-AUC: " + ", ".join(f"{m} {auc_ms(m)[0]:.3f}" for m in HOSTED))
    lines.append("")

    ef = {(r["model"], r["condition"]): r for r in fr if r["org"] == "pooled"}
    lines += ["Results, GUIDE 'Exemplars':",
              f"  exemplar label correct for {ef[('decisions', 'ex')]['exemplar_label_correct_rate']:.3f} of test prompts",
              "  verdict equals the (most similar) exemplar's label, 3-class: " + ", ".join(
                  f"{mm} {cond} {ef[(mm, cond)]['exemplar_following_rate_3class']:.3f}" for (mm, cond) in ef)]
    for m1, m20 in ((DEX, D20), (JEX, J20)):
        a1, a20 = pts[m1][1], pts[m20][1]
        lines.append(f"  {m20}: gate false escalations {a1} -> {a20} ({100 * (a1 - a20) / max(a1, 1):.0f}% fewer than one exemplar)")
    x = next(r for r in nv if r["org"] == "pooled")
    lines.append(f"  neighbor vote (top-1 neighbor label): caught {x['caught']} with {x['false_mal']} false escalations, cost c=10 {x['cost_c10']}")
    lines.append(f"  default verdicts at c=10: RiskFit {fc_cost('RiskFit')}, CE {fc_cost('CE')}")
    lines.append("")

    lines += ["Figure 2 (left) and caption: crossovers c* in Table 2; dotted line = best local vs "
              f"{DEX}: {fmt_c(cx(f'best local (envelope of the six local models) vs {DEX}'))}"]
    for rr in rw_rows:
        lines.append(f"  reweighted to the production class ratio: {rr['local']} vs {rr['hosted']} c* {rr['c_star']:.1f}")
    lines.append("")

    opr = next(r for r in t_rows if r["scope"] == "pooled" and r["method"] == OEX)
    lines += ["Limitations ('Where hosted models are competitive'):",
              f"  {OEX} gate misses {opr['misses_point_idbytes']} (tie order by prompt id), expected {opr['misses_expected_random_ties']:.1f} "
              "under random tie order",
              f"  {D20} gate cost {gcost(D20)} vs RiskFit {gcost('RiskFit')}", ""]
    if pt:
        lines += ["Paired per-organization tests (Wilcoxon, Holm-adjusted within each family):"]
        for r in pt:
            lines.append(f"  {r['set']:30s} {r['comparison']:32s} {r['metric']:22s} mean diff {r['mean_diff']:+.3f} "
                         f"({r['n_pos']}/{r['n_neg']}/{r['n_tie']})  p_holm {r['wilcoxon_p_holm']:.3f}")
    (a.out / "paper_numbers.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
