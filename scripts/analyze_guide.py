#!/usr/bin/env python3
"""All GUIDE statistics in the paper, from local-model predictions and the released per-prompt outputs.

Inputs (local files only, no network):
  --predictions   directory with riskfit/, ce/, lightgbm/ subdirectories of org_<id>.jsonl (scripts/train_guide.py)
  --hosted        directory with decisions_guide_per_prompt.csv, jev_guide_per_prompt.csv, opus_guide_per_prompt.csv
                  (default: data/guide_outputs, the released outputs)
  --five-exemplar Decisions and Jev with five retrieved exemplars (default: the released
                  data/guide_outputs/decisions_jev_5ex_guide_per_prompt.csv; "none" skips both conditions)
  --open-llm      CyberSecQwen-4B zero-shot (default: the released
                  data/guide_outputs/cybersecqwen4b_zeroshot_guide_per_prompt.csv, from scripts/run_cybersecqwen.py;
                  "none" skips it)
  --splits        the splits/ directory (prompt ids, labels and the exemplars' training labels)
Methods: RiskFit, CE, LightGBM; Decisions / Jev / Opus, each zero-shot and with one exemplar; Decisions and Jev
with five exemplars; CyberSecQwen-4B zero-shot. Opus scored test prompts only, so the deployable gate and
validation-tuned thresholds are n/a for it. Adding the five-exemplar and CyberSecQwen conditions does not change
any number of the nine original methods (the bootstrap resamples prompt positions, independently of the methods);
only the "dominated_by" labels of the frontier table can name the new methods.

Outputs (--out): full_coverage, matched_fpr, auc, gate80, gate80_ties, frontier_cost, bootstrap_gate80,
bootstrap_cost_crossovers, bootstrap_min_cost_probability, paired_tests, exemplar_diagnostics,
exemplar_following_test, neighbor_vote, hosted_agreement, crossovers_reweighted (.csv), summary.json, and
paper_numbers.txt (every GUIDE number quoted in the paper, grouped by the table, figure or paragraph that uses it).

Score conventions: local models rank the gate by max class probability and use p(malicious) for ranking;
decision models gate on their own verdict confidence and use the predicate probability (q2; q1 = choice
probability as a sensitivity); Opus gates on its verbalized confidence and ranks by its p_malicious;
CyberSecQwen-4B gates on its largest label probability and ranks by p(malicious).
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
D5, J5, CSQ = "Decisions + 5 exemplars", "Jev + 5 exemplars", "CyberSecQwen-4B zero-shot"
DECISION = [DZS, DEX, JZS, JEX]
OPUS = [OZS, OEX]
FIVE = [D5, J5]           # follow-up conditions, optional (--five-exemplar)
OPEN = [CSQ]              # open LLM run locally, optional (--open-llm); compared like a hosted model
BASE9 = list(LOCAL) + DECISION + OPUS  # the original nine methods (pools named "all 9 methods")
HOSTED = DECISION + OPUS + FIVE + OPEN  # every non-trained comparator (lists are pruned in place in main())
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
    ap.add_argument("--five-exemplar", default=str(ROOT / "data" / "guide_outputs" / "decisions_jev_5ex_guide_per_prompt.csv"))
    ap.add_argument("--open-llm", default=str(ROOT / "data" / "guide_outputs" / "cybersecqwen4b_zeroshot_guide_per_prompt.csv"))
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
    five = {}
    if FIVE:  # Decisions and Jev with five exemplars: loaded exactly like the one-exemplar CSVs above
        for r in csv.DictReader(Path(a.five_exemplar).open(newline="")):
            if r["org"] not in orgs:
                continue
            five[(r["split"], r["prompt_id"])] = r
            for m, pfx in ((D5, "dec5"), (J5, "jev5")):
                ok = r[f"{pfx}_verdict"] not in ("", None)
                rec = {"id": r["prompt_id"], "y": C2X[r["label"]], "pred": C2X[r[f"{pfx}_verdict"]] if ok else "unresolved",
                       "probs": [fnum(r[f"{pfx}_p_fp"]) or 0.0, fnum(r[f"{pfx}_p_benign"]) or 0.0,
                                 fnum(r[f"{pfx}_p_mal_q1"]) or 0.0] if ok else [0.0] * 3,
                       "conf": fnum(r[f"{pfx}_confidence"]) if ok else 0.0, "q2": fnum(r[f"{pfx}_p_mal_q2"]) if ok else 0.0}
                data.setdefault((m, r["org"]), {"test": [], "val": []})[r["split"]].append(rec)
    if OPEN:  # CyberSecQwen-4B: softmax over the three verdict labels' summed log-probabilities
        for r in csv.DictReader(Path(a.open_llm).open(newline="")):
            if r["org"] not in orgs:
                continue
            ps = [float(r["p_fp"]), float(r["p_benign"]), float(r["p_malicious"])]
            assert abs(sum(ps) - 1) < 1e-6 and float(r["confidence"]) == max(ps), r["id"]
            rec = {"id": r["id"], "y": C2X[r["y"]], "pred": C2X[r["verdict"]], "probs": ps,
                   "conf": float(r["confidence"]), "q2": ps[2]}
            data.setdefault((CSQ, r["org"]), {"test": [], "val": []})[r["split"]].append(rec)
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
    return orgs, splits, data, five


def main() -> int:
    a = parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for path, group in ((a.five_exemplar, FIVE), (a.open_llm, OPEN)):  # optional conditions: prune the lists in place
        if path.lower() == "none" or not Path(path).exists():
            for m in list(group):
                for lst in (group, HOSTED, METHODS, HAS_VAL):
                    if m in lst:
                        lst.remove(m)
    orgs, splits, DATA, FIVE_ROWS = load(a)
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
    SCORES = {m: (("p_mal", "q2"),) if m in LOCAL else ((("p_malicious", "q2"),) if m in OPUS + OPEN else (("q1", "q1"), ("q2", "q2"))) for m in METHODS}

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
            pools.append(("all 9 methods", BASE9))
            if FIVE:
                pools.append(("all 9 + both five-exemplar conditions", BASE9 + FIVE))
            if OPEN:
                pools.append(("all 9 + CyberSecQwen-4B", BASE9 + OPEN))
            if FIVE and OPEN:
                pools.append(("all 12 methods", METHODS))
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
                sname = "p_mal" if m in LOCAL else ("p_malicious" if m in OPUS + OPEN else dscore)
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
        # follow-up conditions: separate Holm families, so the families above are unchanged
        extra = []
        if FIVE:
            extra += [("Decisions + 5 exemplars vs local", [(D5, "RiskFit"), (D5, "CE")]),
                      ("Jev + 5 exemplars vs local", [(J5, "RiskFit"), (J5, "CE")]),
                      ("5 exemplars vs 1 exemplar (same model)", [(D5, DEX), (J5, JEX)])]
        if OPEN:
            extra += [("CyberSecQwen-4B vs local", [(CSQ, "RiskFit"), (CSQ, "CE")])]
        for set_name, comps in extra:
            pt += family(set_name, "primary: gate (a), hosted q2", comps, M3, "a", "q2")
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
        if FIVE:  # five exemplars: agreement with the most similar one (= the one-exemplar condition's exemplar)
            for name, pfx in (("decisions", "dec5"), ("jev", "jev5")):
                assert all(FIVE_ROWS[k]["ex1_prompt_id"] == ppd[k]["exemplar_prompt_id"] for k in keys), "top-1 exemplar differs"
                ok = [k for k in keys if FIVE_ROWS[k][f"{pfx}_verdict"]]
                v = {k: C2X[FIVE_ROWS[k][f"{pfx}_verdict"]] for k in ok}
                wrong = [k for k in ok if exl[k] != tl[k]]
                fr.append({"org": o, "model": name, "condition": "5ex (vs most similar exemplar)", "n_resolved": len(ok),
                           "exemplar_label_correct_rate": sum(exl[k] == tl[k] for k in keys) / len(keys),
                           "exemplar_following_rate_3class": sum(v[k] == exl[k] for k in ok) / len(ok) if ok else None,
                           "exemplar_following_rate_binary_mal": sum(ism(v[k]) == ism(exl[k]) for k in ok) / len(ok) if ok else None,
                           "n_exemplar_label_wrong": len(wrong),
                           "follows_wrong_exemplar_rate_3class": sum(v[k] == exl[k] for k in wrong) / len(wrong) if wrong else None})
    write("exemplar_following_test.csv", fr)

    # a vote over labeled neighbors, no model: the most similar training prompt's label, and the binary majority of
    # the five most similar (malicious iff >= 3 of 5); exemplar ids come from the five-exemplar file
    nv = []
    if FIVE:
        for o in orgs + ["pooled"]:
            keys = [k for k in FIVE_ROWS if k[0] == "test" and (o == "pooled" or FIVE_ROWS[k]["org"] == o)]
            tl = {k: C2X[FIVE_ROWS[k]["label"]] for k in keys}
            ex5 = {k: [lab[FIVE_ROWS[k][f"ex{j}_prompt_id"]] for j in range(1, 6)] for k in keys}
            for rule, flag in (("top-1 neighbor label", lambda k: ism(ex5[k][0])),
                               ("majority of 5 neighbors (>= 3 malicious)", lambda k: sum(ism(x) for x in ex5[k]) >= 3)):
                cau = sum(flag(k) for k in keys if ism(tl[k])); nm = sum(ism(tl[k]) for k in keys)
                fm = sum(flag(k) for k in keys if not ism(tl[k])); nn = len(keys) - nm
                nv.append({"org": o, "rule": rule, "caught": cau, "n_mal": nm, "false_mal": fm, "n_nonmal": nn,
                           **{f"cost_c{c}": c * (nm - cau) + fm for c in CS}})
        write("neighbor_vote.csv", nv)

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

    def gcost(kind, m, c=10):
        p = Gt[(kind, m)]["pooled"]; return c * p["misses"] + p["false_esc"]

    def sname_of(m):
        return "p_mal" if m in LOCAL else ("p_malicious" if m in OPUS + OPEN else "q2")

    def mf(m, target, sname=None):
        r = [x for x in mf_rows if x["method"] == m and x["score"] == (sname or sname_of(m)) and x["org"] == "pooled"
             and x["target"] == target and x["selection"] == "validation-tuned"]
        return r[0] if r else None

    def auc_ms(m):
        r = next(r for r in auc_rows if r["method"] == m and r["score"] == sname_of(m) and r.get("metric") == "roc_auc")
        return r["mean"], r["sd"]

    def cxr(pair, kind="a"):
        return next(x for x in cx_rows if x["gate"] == GATES[kind] and x["pair"] == pair)

    def cx(pair, kind="a"):
        r = cxr(pair, kind)
        return math.inf if r["point_region_local_cheaper"] == "always" else r["point_c_star"]

    def fmt_c(v):
        return "inf" if v is None or v == math.inf else f"{v:.1f}"

    def bci(m, kind="a"):
        r = next(x for x in bg_rows if x["gate"] == GATES[kind] and x["method"] == m)
        return f"{r['misses_ci95_low']:.0f}-{r['misses_ci95_high']:.0f}"

    def pool_share(pool, c, who=None):
        rows = [r for r in mc_rows if r["gate"] == GATES["a"] and r["candidate_pool"] == pool and r["c"] == c]
        return sum(r["prob_min_cost_ties_split"] for r in rows if (r["method"] in LOCAL if who is None else r["method"] == who))

    ENV = "best local (envelope of RiskFit/CE/LightGBM) vs "
    shown = [m for m in METHODS]
    lines = [f"GUIDE organizations: {len(orgs)}; test prompts {S['n_test_prompts']} ({nm_g} malicious, {nn_g} non-malicious)",
             "Every line names the place in the paper (draft fix20) that quotes the number.", ""]

    lines += ["Figure 2 (left panel) and RQ1 'Default and tuned operating points': default verdicts, caught / false escalations"]
    for m in shown:
        p = S["full_coverage"][m]
        lines.append(f"  {m:28s} {p['caught']}/{p['false_mal']}  kappa {p['kappa']:.3f}  cost c=10 {10 * (p['n_mal'] - p['caught']) + p['false_mal']}")
    hz = [S["full_coverage"][m] for m in HOSTED if m not in OPEN]
    lines.append(f"  hosted models: caught {min(p['caught'] for p in hz)}-{max(p['caught'] for p in hz)}, "
                 f"false escalations {min(p['false_mal'] for p in hz)}-{max(p['false_mal'] for p in hz)}")
    if OPEN:
        nmal = sum(ism(r["pred"]) for o in orgs for r in DATA[(CSQ, o)]["test"])
        lines.append(f"  CyberSecQwen-4B labeled {nmal} of {S['n_test_prompts']} test prompts malicious")

    targets = [("@RiskFit", "RiskFit default FPR"), ("@1%", "FPR 1%"), ("@2%", "FPR 2%"), ("@5%", "FPR 5%")]
    t3 = [m for m in list(LOCAL) + [DEX, D5, JEX, J5] if m in METHODS]
    lines += ["", "Table 3 (tab:threshold): validation-tuned per-organization threshold, caught / false escalations "
                  "(hosted: predicate probability); * = lowest cost 10*misses+FE in the column"]
    best = {}
    for col, t in targets:
        costs = {m: 10 * (mf(m, t)["n_mal"] - mf(m, t)["tp"]) + mf(m, t)["fp"] for m in t3}
        best[col] = min(costs, key=costs.get)
    for m in t3:
        cells = []
        for col, t in targets:
            r = mf(m, t)
            cells.append(f"{col} {r['tp']}/{r['fp']}{'*' if best[col] == m else ''}")
        lines.append(f"  {m:28s} " + "   ".join(cells))
    lines.append("  RQ1 text: cost at c=10 at RiskFit's FPR: " + ", ".join(
        f"{m} {10 * (mf(m, 'RiskFit default FPR')['n_mal'] - mf(m, 'RiskFit default FPR')['tp']) + mf(m, 'RiskFit default FPR')['fp']}"
        for m in t3))
    lines.append("  zero-shot conditions at RiskFit's FPR (not in the table): " + ", ".join(
        f"{m} {mf(m, 'RiskFit default FPR')['tp']}/{mf(m, 'RiskFit default FPR')['fp']}" for m in (DZS, JZS)))

    lines += ["", "Table 4 (tab:guidegate) and RQ2 'Confidence gating': 80% test-quantile gate"]
    lines.append(f"  {'method':28s} {'miss/FE':>9s}  cost c=10  c* vs RiskFit  c* vs CE   ROC-AUC mean+-sd over orgs")
    for m in shown:
        mu, sd = auc_ms(m)
        cs_ = ("--", "--") if m in LOCAL else (fmt_c(cx(f"RiskFit vs {m}")), fmt_c(cx(f"CE vs {m}")))
        lines.append(f"  {m:28s} {g('a', m):>9s}  {gcost('a', m):9d}  {cs_[0]:>12s}  {cs_[1]:>8s}   {mu:.3f} +- {sd:.3f}")
    lines.append(f"  cost at c=5/10/20: " + "; ".join(f"{m} {gcost('a', m, 5)}/{gcost('a', m, 10)}/{gcost('a', m, 20)}" for m in LOCAL))
    for c in (5, 10, 20):
        lines.append(f"  cheapest point estimate at c={c}: {min(METHODS, key=lambda m: gcost('a', m, c))}")
    lines.append(f"  share of bootstrap replicates in which each is cheapest among all 9 at c=10: RiskFit "
                 f"{pool_share('all 9 methods', 10, 'RiskFit'):.4f}, CE {pool_share('all 9 methods', 10, 'CE'):.4f}")
    for pool in ("all 9 methods", "all 9 + both five-exemplar conditions", "all 9 + CyberSecQwen-4B", "all 12 methods"):
        if any(r["candidate_pool"] == pool for r in mc_rows):
            lines.append(f"  P(a local model is cheapest), pool '{pool}': c=10 {pool_share(pool, 10):.4f}, c=20 {pool_share(pool, 20):.4f}")
    for m in OPUS[1:]:
        lines.append(f"  RiskFit / CE weakly dominate {m} (fewer misses and FE) in {S['share_local_weakly_dominates']['RiskFit vs ' + m]:.3f} / "
                     f"{S['share_local_weakly_dominates']['CE vs ' + m]:.3f} of replicates")
    lines.append(f"  95% bootstrap interval of gate misses: {DEX} {g('a', DEX).split('/')[0]} [{bci(DEX)}], RiskFit "
                 f"{g('a', 'RiskFit').split('/')[0]} [{bci('RiskFit')}]")
    fe_ratio = Gt[("a", DEX)]["pooled"]["false_esc"] / Gt[("a", "RiskFit")]["pooled"]["false_esc"]
    lines.append(f"  {DEX} false escalations / RiskFit's: {fe_ratio:.1f}x")
    r = cxr(f"RiskFit vs {DEX}")
    lines.append(f"  RiskFit vs {DEX}: c* {fmt_c(cx(f'RiskFit vs {DEX}'))}, 95% interval {r['boot_full_p2.5 (inf=local always cheaper)']:.0f} "
                 f"to {fmt_c(r['boot_full_p97.5'])}")
    lines.append(f"  deployable gate, best local vs {DEX}: c* {fmt_c(cx(ENV + DEX, 'b'))}")
    lines += ["  deployable gate, misses / false escalations: " + "; ".join(f"{m} {g('b', m)}" for m in HAS_VAL)]

    lines += ["", "Figure 3 (left panel) and its caption: crossovers c* below which the local model is cheaper (test-quantile gate)"]
    for h in HOSTED:
        lines.append(f"  vs {h:28s} RiskFit {fmt_c(cx('RiskFit vs ' + h))}, CE {fmt_c(cx('CE vs ' + h))}, "
                     f"LightGBM {fmt_c(cx('LightGBM vs ' + h))}, best local {fmt_c(cx(ENV + h))}")
    lines += ["  reweighted to the production class ratio (73:1419):"]
    for rr in rw_rows:
        if rr["analysis"].startswith("GUIDE reweighted") and rr["hosted"] in (DEX, DZS):
            lines.append(f"    {rr['local']:10s} vs {rr['hosted']:22s} c* {rr['c_star']:.1f}  [{rr['boot_lo']:.0f}, {rr['boot_hi']}]")

    ef = {(r["model"], r["condition"]): r for r in fr if r["org"] == "pooled"}
    lines += ["", "RQ2 'Exemplars':",
              f"  exemplar label correct for {ef[('decisions', 'ex')]['exemplar_label_correct_rate']:.3f} of test prompts",
              "  verdict equals the (most similar) exemplar's label, 3-class: " + ", ".join(
                  f"{mm} {cond} {ef[(mm, cond)]['exemplar_following_rate_3class']:.3f}" for (mm, cond) in ef)]
    if FIVE:
        for m1, m5 in ((DEX, D5), (JEX, J5)):
            a1, a5 = Gt[("a", m1)]["pooled"]["false_esc"], Gt[("a", m5)]["pooled"]["false_esc"]
            lines.append(f"  {m5}: gate false escalations {a1} -> {a5} ({100 * (a1 - a5) / a1:.0f}% fewer)")
        nb5 = {(r["org"], r["rule"]): r for r in nv}
        for rule in ("top-1 neighbor label", "majority of 5 neighbors (>= 3 malicious)"):
            x = nb5[("pooled", rule)]
            lines.append(f"  neighbor vote, {rule}: caught {x['caught']} with {x['false_mal']} false escalations, cost c=10 {x['cost_c10']}")
        lines.append("  for comparison, default verdicts at c=10: " + ", ".join(
            f"{m} {10 * (S['full_coverage'][m]['n_mal'] - S['full_coverage'][m]['caught']) + S['full_coverage'][m]['false_mal']}" for m in ("RiskFit", "CE")))

    ha = {(r["pair"], r["condition"]): r for r in ag}
    lines += ["", "Limitations ('Where hosted models are competitive'):",
              f"  Spearman correlation of the two decision models' predicate scores, with exemplar: "
              f"{ha[('decisions vs jev', '+ exemplar')]['spearman_p_malicious']:.3f}",
              "  Opus gate ties at the cutoff: " + ", ".join(
                  f"{r['method']}: misses {r['misses_point_idbytes']} (tie order by prompt id), range {r['misses_min']}-{r['misses_max']}, "
                  f"expected {r['misses_expected_random_ties']:.1f}" for r in t_rows if r["scope"] == "pooled" and r["method"] in OPUS)]
    if pt:
        lines += ["", "Paired per-organization tests (Wilcoxon, Holm-adjusted within each family):"]
        for r in pt:
            if r["family"].startswith(("RQ1", "primary")) and r["metric"] in ("mal_recall", "matched_val_fpr_recall", "matched_fpr_recall", "gate_fe_rate"):
                lines.append(f"  {r['set']:38s} {r['comparison']:42s} {r['metric']:22s} mean diff {r['mean_diff']:+.3f} "
                             f"({r['n_pos']}/{r['n_neg']}/{r['n_tie']})  p_holm {r['wilcoxon_p_holm']:.3f}")
    (a.out / "paper_numbers.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
