#!/usr/bin/env python3
"""All GUIDE statistics in the paper, from local-model predictions and the released hosted-model outputs.

Inputs (local files only, no network):
  --predictions  directory with riskfit/, ce/, lightgbm/ subdirectories of org_<id>.jsonl (scripts/train_guide.py)
  --hosted       directory with decisions_guide_per_prompt.csv, jev_guide_per_prompt.csv, opus_guide_per_prompt.csv
                 (default: data/guide_outputs, the released outputs)
  --splits       the splits/ directory (prompt ids, labels and the exemplars' training labels)
Methods: RiskFit, CE, LightGBM; Decisions / Jev / Opus, each zero-shot and with one exemplar. Opus scored test
prompts only, so the deployable gate and validation-tuned thresholds are n/a for it.

Outputs (--out): full_coverage, matched_fpr, auc, gate80, gate80_ties, frontier_cost, bootstrap_gate80,
bootstrap_cost_crossovers, bootstrap_min_cost_probability, paired_tests, exemplar_diagnostics,
exemplar_following_test, hosted_agreement, crossovers_reweighted (.csv), summary.json, and paper_numbers.txt
(the GUIDE numbers quoted in the paper, with where they appear).

Score conventions: local models rank the gate by max class probability and use p(malicious) for ranking;
decision models gate on their own verdict confidence and use the predicate probability (q2; q1 = choice
probability as a sensitivity); Opus gates on its verbalized confidence and ranks by its p_malicious.
Exact Wilcoxon p-values with zero differences need scipy >= 1.16.

Usage: python scripts/analyze_guide.py --predictions data/predictions --splits data/guide_splits/splits --out results/guide
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
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score  # noqa: E402

from riskfit import evaluation as E  # noqa: E402
from riskfit import guide as G  # noqa: E402

LABELS, MAL, ism = E.LABELS, E.MAL, E.ism
C2X = {"false_positive": LABELS[0], "benign": LABELS[1], "malicious": LABELS[2], "": "unresolved", None: "unresolved"}
LOCAL = {"RiskFit": "riskfit", "CE": "ce", "LightGBM": "lightgbm"}
DZS, DEX, JZS, JEX, OZS, OEX = ("Decisions zero-shot", "Decisions + exemplar", "Jev zero-shot", "Jev + exemplar",
                                "Opus zero-shot", "Opus + exemplar")
DECISION = [DZS, DEX, JZS, JEX]
OPUS = [OZS, OEX]
HOSTED = DECISION + OPUS
METHODS = list(LOCAL) + HOSTED
HAS_VAL = [m for m in METHODS if m not in OPUS]
NA_VAL = "n/a: no validation scores (test prompts only)"
CS = (5, 10, 20, 50)
B, SEED = 2000, 42
GATES = {"a": "(a) per-org test top 80% by confidence", "b": "(b) deployable: tau at 80% validation coverage"}
GM = {"a": METHODS, "b": HAS_VAL}
PROD_MAL, PROD_NONMAL = 73, 1419  # production class ratio for prevalence reweighting


def fnum(x):
    return None if x in (None, "", "None") else float(x)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--hosted", type=Path, default=ROOT / "data" / "guide_outputs")
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--orgs", help="comma-separated subset (smoke runs); default: all 12")
    ap.add_argument("--n-boot", type=int, default=B)
    return ap.parse_args()


# ----------------------------------------------------------------------------- load
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
                ok = r[f"{pfx}_verdict"] not in ("", None)
                rec = {"id": r["prompt_id"], "y": C2X[r["label"]], "pred": C2X[r[f"{pfx}_verdict"]] if ok else "unresolved",
                       "probs": [fnum(r[f"{pfx}_p_fp"]) or 0.0, fnum(r[f"{pfx}_p_benign"]) or 0.0,
                                 fnum(r[f"{pfx}_p_mal_q1"]) or 0.0] if ok else [0.0] * 3,
                       "conf": fnum(r[f"{pfx}_confidence"]) if ok else 0.0, "q2": fnum(r[f"{pfx}_p_mal_q2"]) if ok else 0.0}
                data.setdefault((m, r["org"]), {"test": [], "val": []})[r["split"]].append(rec)
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
    return orgs, splits, data


def main() -> int:
    a = parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    orgs, splits, DATA = load(a)
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

    # --------------------------------------------------------------------- full coverage (RQ1 default verdicts)
    fc_rows, FC = [], {}
    for m in METHODS:
        per = {o: E.full_coverage(DATA[(m, o)]["test"]) for o in orgs}
        FC[m] = per
        fc_rows += [{"method": m, "scope": f"org {o}", **per[o]} for o in orgs]
        pooled = E.full_coverage([r for o in orgs for r in DATA[(m, o)]["test"]])
        fc_rows.append({"method": m, "scope": "pooled (per prompt)", **pooled})
        for k in ("recall", "fpr", "precision", "accuracy", "kappa", "macro_f1"):
            mu, sd, n = E.mean_sd([per[o][k] for o in orgs])
            fc_rows.append({"method": m, "scope": "mean over orgs", "metric": k, "mean": mu, "sd": sd, "n_orgs": n})
        S.setdefault("full_coverage", {})[m] = pooled
    write("full_coverage.csv", fc_rows)

    # --------------------------------------------------------------------- matched FPR + AUC (RQ1 thresholds)
    SCORES = {m: (("p_mal", "q2"),) if m in LOCAL else ((("p_malicious", "q2"),) if m in OPUS else (("q1", "q1"), ("q2", "q2"))) for m in METHODS}

    def score(m, r, key):
        return (r["probs"][2] if key == "q1" else r["q2"]) if m in HOSTED else r["probs"][2]

    mf_rows, auc_rows, MF = [], [], collections.defaultdict(dict)
    for o in orgs:
        targets = [("RiskFit default FPR", FC["RiskFit"][o]["fpr"]), ("FPR 1%", 0.01), ("FPR 2%", 0.02), ("FPR 5%", 0.05)]
        for m in METHODS:
            te, va = DATA[(m, o)]["test"], DATA[(m, o)]["val"]
            for sname, key in SCORES[m]:
                y = [ism(r["y"]) for r in te]
                s = [score(m, r, key) for r in te]
                auc_rows.append({"method": m, "score": sname, "org": o, "roc_auc": float(roc_auc_score(y, s)),
                                 "pr_auc": float(average_precision_score(y, s)), "n": len(te), "n_mal": sum(y),
                                 "n_distinct_scores": len(set(s)), "prevalence": sum(y) / len(y)})
                for tname, f in targets:
                    for sel, src in (("validation-tuned", va), ("TEST-tuned oracle upper bound", te)):
                        if sel == "validation-tuned" and m in OPUS:
                            mf_rows.append({"method": m, "score": sname, "org": o, "target": tname, "target_fpr": f, "selection": sel, "note": NA_VAL})
                            continue
                        thr = E.np_threshold([score(m, r, key) for r in src if not ism(r["y"])], f)
                        row = {"method": m, "score": sname, "org": o, "target": tname, "target_fpr": f, "selection": sel,
                               **E.at_threshold(te, lambda r: score(m, r, key), thr)}
                        mf_rows.append(row)
                        MF[(m, sname, tname, sel)][o] = row
    for (m, sname, tname, sel), per in list(MF.items()):
        tp = sum(r["tp"] for r in per.values()); npos = sum(r["n_mal"] for r in per.values())
        fp = sum(r["fp"] for r in per.values()); nneg = sum(r["n_neg"] for r in per.values())
        mu, sd, _ = E.mean_sd([r["recall"] for r in per.values()])
        mf_rows.append({"method": m, "score": sname, "org": "pooled", "target": tname, "selection": sel, "tp": tp, "n_mal": npos,
                        "fp": fp, "n_neg": nneg, "recall": tp / npos, "fpr": fp / nneg, "mean_recall_over_orgs": mu, "sd_recall_over_orgs": sd})
    write("matched_fpr.csv", mf_rows)
    for m in METHODS:
        for sname, _ in SCORES[m]:
            rr = [r for r in auc_rows if r["method"] == m and r["score"] == sname]
            for k in ("roc_auc", "pr_auc"):
                mu, sd, n = E.mean_sd([r[k] for r in rr])
                auc_rows.append({"method": m, "score": sname, "org": "mean over orgs", "metric": k, "mean": mu, "sd": sd, "n_orgs": n})
            allte = [r for o in orgs for r in DATA[(m, o)]["test"]]
            y = [ism(r["y"]) for r in allte]; s = [score(m, r, sname) for r in allte]
            auc_rows.append({"method": m, "score": sname, "org": "pooled (all test prompts)", "roc_auc": float(roc_auc_score(y, s)),
                             "pr_auc": float(average_precision_score(y, s)), "n": len(y), "n_mal": sum(y)})
    write("auc.csv", auc_rows)

    # --------------------------------------------------------------------- 80% gates (RQ2)
    g_rows, Gt = [], {}
    for kind, gname in GATES.items():
        for m in METHODS:
            if m not in GM[kind]:
                g_rows.append({"gate": gname, "method": m, "scope": "pooled", "note": NA_VAL})
                continue
            per = {o: E.gate(DATA[(m, o)]["test"], kind, DATA[(m, o)]["val"]) for o in orgs}
            Gt[(kind, m)] = per
            g_rows += [{"gate": gname, "method": m, "scope": f"org {o}", **per[o]} for o in orgs]
            tot = {k: sum(per[o][k] for o in orgs) for k in ("kept", "n", "n_mal", "n_nonmal", "misses", "false_esc", "mal_routed")}
            tot.update({"coverage": tot["kept"] / tot["n"], "system_recall": 1 - tot["misses"] / tot["n_mal"],
                        "fe_per_1k_nonmal": 1000 * tot["false_esc"] / tot["n_nonmal"]})
            Gt[(kind, m)]["pooled"] = tot
            g_rows.append({"gate": gname, "method": m, "scope": "pooled", **tot})
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

    fr_rows = []
    for kind, gname in GATES.items():
        pts = {m: (Gt[(kind, m)]["pooled"]["misses"], Gt[(kind, m)]["pooled"]["false_esc"]) for m in GM[kind]}
        for m in GM[kind]:
            dom = E.dominated_by(pts, m)
            row = {"gate": gname, "method": m, "misses": pts[m][0], "false_esc": pts[m][1],
                   "coverage": Gt[(kind, m)]["pooled"]["coverage"], "on_frontier": not dom, "dominated_by": "; ".join(dom)}
            row.update({f"cost_c{c}": c * pts[m][0] + pts[m][1] for c in CS})
            fr_rows.append(row)
        for c in CS:
            S.setdefault("cheapest_point", {})[f"{kind}_c{c}"] = min(GM[kind], key=lambda m: c * pts[m][0] + pts[m][1])
    write("frontier_cost.csv", fr_rows)

    # --------------------------------------------------------------------- bootstrap
    MISS, FE = E.bootstrap_gate_counts(DATA, orgs, METHODS, set(HAS_VAL), n_boot=nb, seed=SEED, ref_method="RiskFit")
    bg_rows = []
    for kind, gname in GATES.items():
        for m in GM[kind]:
            p = Gt[(kind, m)]["pooled"]
            ml, mh = E.pct95(MISS[(kind, m)].sum(1)); fl, fh = E.pct95(FE[(kind, m)].sum(1))
            bg_rows.append({"gate": gname, "method": m, "point_misses": p["misses"], "misses_ci95_low": ml, "misses_ci95_high": mh,
                            "point_false_esc": p["false_esc"], "false_esc_ci95_low": fl, "false_esc_ci95_high": fh})
    write("bootstrap_gate80.csv", bg_rows)

    cx_rows, mc_rows = [], []
    for kind, gname in GATES.items():
        for dm in [h for h in HOSTED if h in GM[kind]]:
            Md, Fd = MISS[(kind, dm)].sum(1), FE[(kind, dm)].sum(1)
            pd_ = Gt[(kind, dm)]["pooled"]
            for lm in LOCAL:
                pl = Gt[(kind, lm)]["pooled"]
                kd, cpt = E.region_local_cheaper(pl["misses"], pl["false_esc"], pd_["misses"], pd_["false_esc"])
                cx_rows.append({"gate": gname, "pair": f"{lm} vs {dm}", "point_misses_local": pl["misses"], "point_fe_local": pl["false_esc"],
                                "point_misses_hosted": pd_["misses"], "point_fe_hosted": pd_["false_esc"],
                                "point_region_local_cheaper": kd, "point_c_star": cpt,
                                **E.crossover_bootstrap(MISS[(kind, lm)].sum(1), FE[(kind, lm)].sum(1), Md, Fd)})
            Ml = np.stack([MISS[(kind, lm)].sum(1) for lm in LOCAL], 1); Fl = np.stack([FE[(kind, lm)].sum(1) for lm in LOCAL], 1)
            pcs = [E.region_local_cheaper(Gt[(kind, lm)]["pooled"]["misses"], Gt[(kind, lm)]["pooled"]["false_esc"], pd_["misses"], pd_["false_esc"]) for lm in LOCAL]
            pt_env = max([c for k, c in pcs if k == "below"], default=None)
            arg = [lm for lm, (k, c) in zip(LOCAL, pcs) if k == "below" and c == pt_env]
            dom = [lm for lm, (k, c) in zip(LOCAL, pcs) if k == "always"]
            if dom:
                pt_env, arg = math.inf, [f"{x} (dominates: cheaper at every c)" for x in dom]
            cx_rows.append({"gate": gname, "pair": f"best local (envelope of RiskFit/CE/LightGBM) vs {dm}", "point_c_star": pt_env,
                            "point_region_local_cheaper": "always" if dom else ("below" if pt_env else "never"),
                            "point_note": f"attained by {', '.join(arg)}" if arg else "", **E.envelope_bootstrap(Ml, Fl, Md, Fd, nb)})
        pools = []
        if kind == "a":
            pools.append(("all 9 methods", METHODS))
        pools += [("3 local + 4 decision-model conditions", list(LOCAL) + DECISION),
                  ("3 local + both Decisions", list(LOCAL) + [DZS, DEX])]
        pools += [(f"3 local + {h} only", list(LOCAL) + [h]) for h in HOSTED if h in GM[kind]]
        if kind == "a":
            pools += [("3 local + all three exemplar conditions", list(LOCAL) + [DEX, JEX, OEX]),
                      ("3 local + all three zero-shot conditions", list(LOCAL) + [DZS, JZS, OZS])]
        for pool_name, pool in pools:
            for c in CS:
                for m, s in zip(pool, E.min_cost_probability(MISS, FE, kind, pool, c)):
                    p = Gt[(kind, m)]["pooled"]
                    mc_rows.append({"gate": gname, "candidate_pool": pool_name, "c": c, "method": m,
                                    "point_cost": c * p["misses"] + p["false_esc"], "prob_min_cost_ties_split": float(s)})
    write("bootstrap_cost_crossovers.csv", cx_rows)
    write("bootstrap_min_cost_probability.csv", mc_rows)

    # pairwise "local cheaper than hosted at c" shares (paper: RiskFit/CE vs Jev/Opus with exemplars)
    for lm in ("RiskFit", "CE"):
        for h in HOSTED:
            for c in CS:
                cl = c * MISS[("a", lm)].sum(1) + FE[("a", lm)].sum(1); ch = c * MISS[("a", h)].sum(1) + FE[("a", h)].sum(1)
                S.setdefault("share_local_cheaper", {})[f"{lm} vs {h} at c={c}"] = float((cl < ch).mean())
            dom = (MISS[("a", lm)].sum(1) <= MISS[("a", h)].sum(1)) & (FE[("a", lm)].sum(1) <= FE[("a", h)].sum(1))
            S.setdefault("share_local_weakly_dominates", {})[f"{lm} vs {h}"] = float(dom.mean())

    # --------------------------------------------------------------------- paired tests over organizations (Holm)
    def org_metric(m, metric, kind, dscore):
        out = []
        for o in orgs:
            if metric.startswith("matched_fpr_recall"):
                sname = "p_mal" if m in LOCAL else ("p_malicious" if m in OPUS else dscore)
                sel = "validation-tuned" if metric == "matched_fpr_recall" else "TEST-tuned oracle upper bound"
                out.append(MF[(m, sname, "RiskFit default FPR", sel)][o]["recall"])
            elif metric == "gate_miss_rate":
                g = Gt[(kind, m)][o]; out.append(g["misses"] / g["n_mal"])
            elif metric == "gate_fe_rate":
                g = Gt[(kind, m)][o]; out.append(g["false_esc"] / g["n_nonmal"])
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

    def family(set_name, fam, comps, metrics, kind, dscore):
        rows = []
        for hm, ref in comps:
            for metric in metrics:
                x = org_metric(hm, metric, kind, dscore); y = org_metric(ref, metric, kind, dscore)
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
                     [("RiskFit", "CE"), ("RiskFit", "LightGBM")], ("mal_recall", "matched_val_fpr_recall", "kappa"), "a", "q2")
        M3 = ("matched_fpr_recall", "gate_miss_rate", "gate_fe_rate")
        fams = {"primary: gate (a), hosted q2": ("a", "q2"), "sensitivity: gate (b) deployable, hosted q2": ("b", "q2"),
                "sensitivity: gate (a), hosted q1 choice": ("a", "q1")}
        for set_name, comps in (("Decisions + exemplar vs local", [(DEX, "RiskFit"), (DEX, "CE")]),
                                ("Jev + exemplar vs local", [(JEX, "RiskFit"), (JEX, "CE")]),
                                ("Jev vs Decisions (same condition)", [(JEX, DEX), (JZS, DZS)])):
            for fam, (kind, dscore) in fams.items():
                pt += family(set_name, fam, comps, M3, kind, dscore)
        for set_name, comps in (("Opus + exemplar vs local", [(OEX, "RiskFit"), (OEX, "CE")]),
                                ("Opus zero-shot vs local", [(OZS, "RiskFit"), (OZS, "CE")]),
                                ("Opus vs Jev (same condition)", [(OEX, JEX), (OZS, JZS)]),
                                ("Opus vs Decisions (same condition)", [(OEX, DEX), (OZS, DZS)])):
            pt += family(set_name, "primary: gate (a) miss/FE rates", comps, ("gate_miss_rate", "gate_fe_rate"), "a", "q2")
            pt += family(set_name, "sensitivity: TEST-tuned matched-FPR recall", comps, ("matched_fpr_recall_test_oracle",), "a", "q2")
        write("paired_tests.csv", pt)

    # --------------------------------------------------------------------- exemplar use and hosted agreement
    rows_h = {}
    for fname, name in (("decisions_guide_per_prompt.csv", "decisions"), ("jev_guide_per_prompt.csv", "jev")):
        rows_h[name] = {(r["split"], r["prompt_id"]): r for r in csv.DictReader((a.hosted / fname).open(newline="")) if r["org"] in orgs}
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
        for name, pp in (("decisions", rows_h["decisions"]), ("jev", rows_h["jev"]), ("opus", ppo)):
            for pfx in ("zs", "ex"):
                ok = [k for k in keys if pp[k][f"{pfx}_verdict"]]
                wrong = [k for k in ok if exl[k] != tl[k]]
                fr.append({"org": o, "model": name, "condition": pfx, "n_resolved": len(ok),
                           "exemplar_label_correct_rate": sum(exl[k] == tl[k] for k in keys) / len(keys),
                           "exemplar_following_rate_3class": sum(C2X[pp[k][f"{pfx}_verdict"]] == exl[k] for k in ok) / len(ok) if ok else None,
                           "exemplar_following_rate_binary_mal": sum(ism(C2X[pp[k][f"{pfx}_verdict"]]) == ism(exl[k]) for k in ok) / len(ok) if ok else None,
                           "n_exemplar_label_wrong": len(wrong),
                           "follows_wrong_exemplar_rate_3class": sum(C2X[pp[k][f"{pfx}_verdict"]] == exl[k] for k in wrong) / len(wrong) if wrong else None})
    write("exemplar_following_test.csv", fr)

    ag = []
    for na, nb_, pa_, pb_ in (("decisions", "jev", rows_h["decisions"], rows_h["jev"]), ("opus", "jev", ppo, rows_h["jev"]),
                              ("opus", "decisions", ppo, rows_h["decisions"])):
        keys = [k for k in ppd if k[0] == "test"]
        for pfx, cond in (("zs", "zero-shot"), ("ex", "+ exemplar")):
            ok = [k for k in keys if pa_[k][f"{pfx}_verdict"] and pb_[k][f"{pfx}_verdict"]]
            va = [C2X[pa_[k][f"{pfx}_verdict"]] for k in ok]; vb = [C2X[pb_[k][f"{pfx}_verdict"]] for k in ok]
            if na == "decisions":
                sa = [fnum(pa_[k][f"{pfx}_p_mal_q2"]) for k in ok]
            else:
                m_ = OZS if pfx == "zs" else OEX
                po = {r["id"]: r["q2"] for oo in orgs for r in DATA[(m_, oo)]["test"]}
                sa = [po[k[1]] for k in ok]
            sb = [fnum(pb_[k][f"{pfx}_p_mal_q2"]) for k in ok]
            ag.append({"pair": f"{na} vs {nb_}", "split": "test", "condition": cond, "n": len(ok),
                       "exact_3class_agreement": sum(x == y for x, y in zip(va, vb)) / len(ok),
                       "kappa_3class": float(cohen_kappa_score(va, vb, labels=list(LABELS))),
                       "binary_mal_agreement": sum(ism(x) == ism(y) for x, y in zip(va, vb)) / len(ok),
                       "spearman_p_malicious": spearmanr(sa, sb).correlation if len(set(sa)) > 1 and len(set(sb)) > 1 else None})
    write("hosted_agreement.csv", ag)

    # --------------------------------------------------------------------- crossovers reweighted to production prevalence
    nm_g = sum(ism(r["y"]) for o in orgs for r in DATA[("RiskFit", o)]["test"])
    nn_g = sum(not ism(r["y"]) for o in orgs for r in DATA[("RiskFit", o)]["test"])
    w_prod = E.reweight_factor(PROD_MAL, PROD_NONMAL, nm_g, nn_g)
    rng = np.random.default_rng(SEED)
    sizes = {o: len(DATA[("RiskFit", o)]["test"]) for o in orgs}

    def pooled_w(m, w, idx=None):
        mi = fe = 0.0
        for o in orgs:
            te = DATA[(m, o)]["test"]
            if idx is not None:
                te = [te[k] for k in idx[o]]
            x, y = E.weighted_gate(te, w); mi += x; fe += y
        return mi, fe

    rw_rows = []
    for w, tag in ((1.0, "GUIDE unweighted"), (w_prod, "GUIDE reweighted to production class ratio (73:1419)")):
        pts = {m: pooled_w(m, w) for m in list(LOCAL) + HOSTED}
        boots = collections.defaultdict(list)
        for _ in range(nb):
            idx = {o: rng.integers(0, sizes[o], sizes[o]) for o in orgs}
            pb = {m: pooled_w(m, w, idx) for m in list(LOCAL) + HOSTED}
            for h in HOSTED:
                for L in list(LOCAL) + ["best local"]:
                    boots[(L, h)].append(max(E.cstar(*pb[x], *pb[h]) for x in LOCAL) if L == "best local" else E.cstar(*pb[L], *pb[h]))
        for h in HOSTED:
            for L in list(LOCAL) + ["best local"]:
                c = max(E.cstar(*pts[x], *pts[h]) for x in LOCAL) if L == "best local" else E.cstar(*pts[L], *pts[h])
                bs = np.array(boots[(L, h)])
                with np.errstate(invalid="ignore"):  # quantiles between +inf values are fixed up below
                    lo, hi = (float(np.quantile(np.where(np.isfinite(bs), bs, np.inf), q)) for q in (0.025, 0.975))
                row = {"analysis": tag, "local": L, "hosted": h, "weight_nonmal": w, "hosted_misses": pts[h][0],
                       "hosted_fe": pts[h][1], "c_star": c, "boot_lo": lo, "boot_hi": hi,
                       "share_infinite": float(np.mean(~np.isfinite(bs)))}
                if row["share_infinite"] > 0.025:
                    row["boot_hi"] = math.inf
                if row["share_infinite"] > 0.975:
                    row["boot_lo"] = math.inf
                if L != "best local":
                    row.update({"local_misses": pts[L][0], "local_fe": pts[L][1]})
                rw_rows.append(row)
    write("crossovers_reweighted.csv", rw_rows)
    S["reweight_factor_nonmal"] = w_prod
    (a.out / "summary.json").write_text(json.dumps(S, indent=1, default=str))

    # --------------------------------------------------------------------- the paper's GUIDE numbers
    def g(kind, m):
        p = Gt[(kind, m)]["pooled"]; return f"{p['misses']}/{p['false_esc']}"

    def mfp(m, sname):
        r = [x for x in mf_rows if x["method"] == m and x["score"] == sname and x["org"] == "pooled"
             and x["target"] == "RiskFit default FPR" and x["selection"] == "validation-tuned"]
        return f"{r[0]['tp']}/{r[0]['fp']}" if r else "n/a"

    def auc_mean(m, sname):
        return next(r["mean"] for r in auc_rows if r["method"] == m and r["score"] == sname and r.get("metric") == "roc_auc")

    def cx(pair_prefix, kind="a"):
        r = next(x for x in cx_rows if x["gate"] == GATES[kind] and x["pair"].startswith(pair_prefix))
        return r["point_c_star"] if r["point_region_local_cheaper"] != "always" else math.inf

    lines = [f"GUIDE organizations: {len(orgs)}; test prompts {S['n_test_prompts']} ({nm_g} malicious, {nn_g} non-malicious)", "",
             "RQ1 default verdicts, caught / false malicious (Results, RQ1: Default operating points):"]
    for m in METHODS:
        p = S["full_coverage"][m]; lines.append(f"  {m:22s} {p['caught']}/{p['false_mal']}  kappa {p['kappa']:.3f}")
    lines += ["", "RQ1 thresholds at RiskFit's default FPR, validation-tuned, caught / false (RQ1: Thresholds and ranking):"]
    lines += [f"  {m:22s} {mfp(m, 'p_mal' if m in LOCAL else 'q2')}" for m in list(LOCAL) + DECISION]
    lines += ["", "Mean per-organization ROC-AUC of p(malicious):"]
    lines += [f"  {m:22s} {auc_mean(m, 'p_mal' if m in LOCAL else ('p_malicious' if m in OPUS else 'q2')):.3f}" for m in METHODS]
    lines += ["", "RQ2 80% test-quantile gate, misses / false escalations (RQ2: Gating on public data):"]
    lines += [f"  {m:22s} {g('a', m)}   cost at c=10: {10 * Gt[('a', m)]['pooled']['misses'] + Gt[('a', m)]['pooled']['false_esc']}" for m in METHODS]
    lines += ["", "Deployable gate, misses / false escalations:"] + [f"  {m:22s} {g('b', m)}" for m in HAS_VAL]
    lines += ["", "Gate ties at the cutoff (Opus): " + ", ".join(
        f"{r['method']}: {r['n_tied_at_tau']} tied, misses {r['misses_min']}-{r['misses_max']}" for r in t_rows
        if r["scope"] == "pooled" and r["method"] in OPUS)]
    allpool = [r for r in mc_rows if r["gate"] == GATES["a"] and r["candidate_pool"] == "all 9 methods"]
    for c in (10, 20):
        share = sum(r["prob_min_cost_ties_split"] for r in allpool if r["c"] == c and r["method"] in LOCAL)
        lines.append(f"P(a local model is cheapest among all 9) at c={c}: {share:.4f}")
    lines += ["", "Crossovers c* (Table: crossover; GUIDE column, test-quantile gate):"]
    for h in (DEX, DZS, JEX, JZS, OEX, OZS):
        lines.append(f"  vs {h:22s} RiskFit {cx('RiskFit vs ' + h)}, CE {cx('CE vs ' + h)}, LightGBM {cx('LightGBM vs ' + h)}, "
                     f"best local {cx('best local (envelope of RiskFit/CE/LightGBM) vs ' + h)}")
    lines.append(f"  deployable gate, best local vs {DEX}: {cx('best local (envelope of RiskFit/CE/LightGBM) vs ' + DEX, 'b')}")
    lines += ["", "Reweighted to the production class ratio (parentheses in Table: crossover):"]
    for r in rw_rows:
        if r["analysis"].startswith("GUIDE reweighted") and r["hosted"] in (DEX, DZS):
            lines.append(f"  {r['local']:10s} vs {r['hosted']:22s} c* {r['c_star']:.1f}  [{r['boot_lo']:.0f}, {r['boot_hi']}]")
    lines += ["", "Shares of bootstrap replicates in which the local model is cheaper at every c (weakly dominates):"]
    lines += [f"  {k}: {v:.3f}" for k, v in S["share_local_weakly_dominates"].items() if k.endswith(("exemplar",))]
    ef = {(r["model"], r["condition"]): r for r in fr if r["org"] == "pooled"}
    lines += ["", f"Exemplar label correct for {ef[('decisions', 'ex')]['exemplar_label_correct_rate']:.2f} of test prompts; "
              f"followed by Decisions {ef[('decisions', 'ex')]['exemplar_following_rate_3class']:.2f}, "
              f"Jev {ef[('jev', 'ex')]['exemplar_following_rate_3class']:.2f}, Opus {ef[('opus', 'ex')]['exemplar_following_rate_3class']:.2f}"]
    if pt:
        lines += ["", "Paired per-organization tests (Wilcoxon, Holm-adjusted p):"]
        for r in pt:
            if r["family"].startswith(("RQ1", "primary")) and r["metric"] in ("mal_recall", "matched_val_fpr_recall", "matched_fpr_recall", "gate_fe_rate"):
                lines.append(f"  {r['set']:34s} {r['comparison']:30s} {r['metric']:20s} mean diff {r['mean_diff']:+.3f} "
                             f"({r['n_pos']}/{r['n_neg']}/{r['n_tie']})  p_holm {r['wilcoxon_p_holm']:.3f}")
    (a.out / "paper_numbers.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
