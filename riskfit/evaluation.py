"""Operating points, confidence gates, cost, crossovers, bootstrap, paired tests and prevalence reweighting.

A prediction record is a dict ``{"id", "y", "pred", "probs": [p_fp, p_benign, p_mal], "conf", "q2"}`` with
labels in LABELS (``pred`` may also be "unresolved" for a hosted call that returned nothing). ``conf`` is the
score the gate ranks by (max class probability, or the hosted model's own confidence); ``probs[2]`` / ``q2``
are p(malicious) scores used for ROC/PR-AUC and validation-tuned thresholds.

Definitions (as in the paper):

* miss: an automated (not routed) malicious alert whose verdict is not malicious; routed alerts are
  assumed handled correctly. false escalation (FE): an automated non-malicious alert labelled malicious.
* cost(c) = c * misses + FE.
* test-quantile gate: per organization/tenant, automate the ceil(0.8 n) most confident test items, ranked
  by (-conf, id bytes); ties at the cutoff are broken by id bytes.
* deployable gate: tau = confidence of the item at the 80% mark of the validation ranking; automate test
  items with conf >= tau.
* crossover c*: for a local model L and a hosted model H at the gate, L is cheaper for c < c* =
  (FE_H - FE_L) / (misses_L - misses_H) when L has more misses; L dominates (c* = inf) when it has no
  more misses and fewer FE.
* bootstrap: resample test items within organization (2,000 replicates, numpy default_rng(seed=42)),
  paired across methods; the gate keeps exactly ceil(0.8 n) resampled items in each replicate.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

LABELS = ("False Positive", "True Positive - Benign", "True Positive - Malicious")
MAL = LABELS[2]


def ism(label: str) -> bool:
    return label == MAL


def ranked(recs: Iterable[dict]) -> list[dict]:
    return sorted(recs, key=lambda r: (-r["conf"], r["id"].encode("utf-8")))


def n_keep(coverage: float, n: int) -> int:
    return min(n, math.ceil(round(coverage * n, 9)))


def np_threshold(neg_scores: Sequence[float], target_fpr: float) -> float:
    """Neyman-Pearson threshold: items with score > t are flagged; at most ``target_fpr`` of negatives exceed t."""
    s = sorted(neg_scores, reverse=True)
    m = math.floor(round(target_fpr * len(s), 9))
    return -1.0 if m >= len(s) else s[m]


def mean_sd(values):
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return (float(np.mean(v)) if v else None, float(np.std(v, ddof=1)) if len(v) > 1 else None, len(v))


# --------------------------------------------------------------------------- full coverage


def full_coverage(recs: list[dict]) -> dict[str, Any]:
    from sklearn.metrics import cohen_kappa_score, f1_score

    y = [r["y"] for r in recs]
    p = [r["pred"] for r in recs]
    tp = sum(ism(a) and ism(b) for a, b in zip(y, p))
    pos = sum(ism(a) for a in y)
    pp = sum(ism(b) for b in p)
    neg, fp = len(y) - pos, pp - tp
    return {"n": len(y), "n_mal": pos, "n_nonmal": neg, "caught": tp, "false_mal": fp,
            "recall": tp / pos if pos else None, "fpr": fp / neg if neg else None, "precision": tp / pp if pp else None,
            "accuracy": sum(a == b for a, b in zip(y, p)) / len(y),
            "kappa": float(cohen_kappa_score(y, p, labels=list(LABELS))) if len(set(y) | set(p)) > 1 else None,
            "macro_f1": float(f1_score(y, p, labels=list(LABELS), average="macro", zero_division=0))}


def at_threshold(test: list[dict], score, thr: float) -> dict[str, Any]:
    pos = [r for r in test if ism(r["y"])]
    neg = [r for r in test if not ism(r["y"])]
    tp = sum(score(r) > thr for r in pos)
    fp = sum(score(r) > thr for r in neg)
    return {"threshold": thr, "tp": tp, "n_mal": len(pos), "fp": fp, "n_neg": len(neg),
            "recall": tp / len(pos), "fpr": fp / len(neg)}


def bayes_threshold(c: float) -> float:
    """Validation-free decision rule: flag malicious iff p(malicious) >= 1 / (1 + c)."""
    return 1.0 / (1.0 + c)


# --------------------------------------------------------------------------- gates


def tau_validation(val: list[dict], coverage: float = 0.8) -> float:
    rv = ranked(val)
    return rv[n_keep(coverage, len(rv)) - 1]["conf"]


def gate(test: list[dict], kind: str = "a", val: list[dict] | None = None, coverage: float = 0.8) -> dict[str, Any]:
    """kind 'a' = test-quantile gate, 'b' = deployable gate (tau from validation)."""
    if kind == "a":
        rk = ranked(test)
        kept = rk[:n_keep(coverage, len(rk))]
        tau = kept[-1]["conf"]
    else:
        tau = tau_validation(val, coverage)
        kept = [r for r in test if r["conf"] >= tau]
    nm = sum(ism(r["y"]) for r in test)
    nn = len(test) - nm
    miss = sum(ism(r["y"]) and not ism(r["pred"]) for r in kept)
    fe = sum((not ism(r["y"])) and ism(r["pred"]) for r in kept)
    return {"kept": len(kept), "n": len(test), "coverage": len(kept) / len(test), "n_mal": nm, "n_nonmal": nn,
            "misses": miss, "false_esc": fe, "mal_routed": nm - sum(ism(r["y"]) for r in kept),
            "system_recall": 1 - miss / nm if nm else None, "fe_per_1k_nonmal": 1000 * fe / nn if nn else None,
            "tau": tau, "test_tied_at_tau": sum(r["conf"] == tau for r in test),
            "n_distinct_conf_test": len({r["conf"] for r in test})}


def gate_ties(test: list[dict], coverage: float = 0.8) -> dict[str, Any]:
    """Range and expectation of gate misses/FE over all tie-breaks at the test-quantile cutoff."""
    rk = ranked(test)
    kk = n_keep(coverage, len(rk))
    tau = rk[kk - 1]["conf"]
    above = [r for r in test if r["conf"] > tau]
    tied = [r for r in test if r["conf"] == tau]
    k_t = kk - len(above)

    def mi(rs):
        return sum(ism(r["y"]) and not ism(r["pred"]) for r in rs)

    def fe(rs):
        return sum((not ism(r["y"])) and ism(r["pred"]) for r in rs)

    ma, fa, mt, ft = mi(above), fe(above), mi(tied), fe(tied)
    return {"n": len(test), "kk": kk, "tau": tau, "n_above_tau": len(above), "n_tied_at_tau": len(tied),
            "n_tied_kept": k_t, "n_tied_routed": len(tied) - k_t, "tied_misses_if_kept": mt, "tied_fe_if_kept": ft,
            "misses_point_idbytes": mi(rk[:kk]), "misses_min": ma + max(0, k_t - (len(tied) - mt)),
            "misses_max": ma + min(k_t, mt), "misses_expected_random_ties": ma + k_t * mt / len(tied),
            "fe_point_idbytes": fe(rk[:kk]), "fe_min": fa + max(0, k_t - (len(tied) - ft)), "fe_max": fa + min(k_t, ft),
            "fe_expected_random_ties": fa + k_t * ft / len(tied), "n_distinct_conf_test": len({r["conf"] for r in test})}


def dominated_by(points: dict[str, tuple[float, float]], m: str) -> list[str]:
    mi, fi = points[m]
    return [n for n in points if n != m and points[n][0] <= mi and points[n][1] <= fi and points[n] != points[m]]


def region_local_cheaper(m_local, fe_local, m_hosted, fe_hosted):
    """Describe {c >= 0 : c*m_local + fe_local < c*m_hosted + fe_hosted} as (kind, c*)."""
    if m_local == m_hosted:
        return ("always" if fe_local < fe_hosted else "never"), None
    cstar = (fe_hosted - fe_local) / (m_local - m_hosted)
    if m_local > m_hosted:
        return ("below", cstar) if cstar > 0 else ("never", cstar)
    return ("always", cstar) if cstar <= 0 else ("above", cstar)


def cstar(m_local, fe_local, m_hosted, fe_hosted) -> float:
    return math.inf if m_local <= m_hosted else (fe_hosted - fe_local) / (m_local - m_hosted)


# --------------------------------------------------------------------------- bootstrap


def bootstrap_gate_counts(data: dict, orgs: list[str], methods: list[str], deployable: set[str],
                          n_boot: int = 2000, seed: int = 42, coverage: float = 0.8, ref_method: str | None = None):
    """Paired bootstrap of pooled gate misses/FE. ``data[(method, org)] = {'test': [...], 'val': [...]}``.

    Returns MISS[(kind, method)] and FE[(kind, method)] arrays of shape (n_boot, n_orgs)."""
    ref = ref_method or methods[0]
    rng = np.random.default_rng(seed)
    prep = {}
    for o in orgs:
        ids = [r["id"] for r in data[(ref, o)]["test"]]
        pos = {i: j for j, i in enumerate(ids)}
        n = len(ids)
        for m in methods:
            rk = ranked(data[(m, o)]["test"])
            order = np.array([pos[r["id"]] for r in rk])
            miss = np.array([ism(r["y"]) and not ism(r["pred"]) for r in rk], float)
            fe = np.array([(not ism(r["y"])) and ism(r["pred"]) for r in rk], float)
            keepb = None
            if m in deployable:
                tau = tau_validation(data[(m, o)]["val"], coverage)
                keepb = np.array([r["conf"] >= tau for r in rk], float)
            prep[(m, o)] = (order, miss, fe, keepb, n_keep(coverage, n), n)
    kinds = ("a", "b")
    MISS = {(k, m): np.zeros((n_boot, len(orgs))) for k in kinds for m in methods if k == "a" or m in deployable}
    FE = {(k, m): np.zeros((n_boot, len(orgs))) for k in kinds for m in methods if k == "a" or m in deployable}
    for b in range(n_boot):
        for oi, o in enumerate(orgs):
            n = prep[(ref, o)][5]
            counts = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
            for m in methods:
                order, miss, fe, keepb, kk, _ = prep[(m, o)]
                cs = counts[order]
                cum = np.cumsum(cs)
                take = np.clip(kk - (cum - cs), 0, cs)
                MISS[("a", m)][b, oi] = take @ miss
                FE[("a", m)][b, oi] = take @ fe
                if keepb is not None:
                    tb = cs * keepb
                    MISS[("b", m)][b, oi] = tb @ miss
                    FE[("b", m)][b, oi] = tb @ fe
    return MISS, FE


def pct95(a) -> tuple[float, float]:
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def crossover_bootstrap(Ml, Fl, Md, Fd) -> dict[str, Any]:
    """Bootstrap distribution of c* for one local/hosted pair (arrays of pooled counts per replicate)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        cs = np.where(Ml != Md, (Fd - Fl) / (Ml - Md), np.nan)
    below = (Ml > Md) & (cs > 0)
    defined = cs[below]
    local_always = ((Ml < Md) & (Fl <= Fd)) | ((Ml == Md) & (Fl < Fd))
    hosted_always = ((Ml >= Md) & (Fl >= Fd)) & ~((Ml == Md) & (Fl == Fd))
    other = ~(below | local_always | hosted_always)
    full = np.where(below, cs, np.where(local_always, np.inf, 0.0))
    return {"boot_share_local_cheaper_below_cstar": float(below.mean()),
            "boot_c_star_p2.5": float(np.percentile(defined, 2.5)) if defined.size else None,
            "boot_c_star_median": float(np.median(defined)) if defined.size else None,
            "boot_c_star_p97.5": float(np.percentile(defined, 97.5)) if defined.size else None,
            "boot_share_local_cheaper_at_all_c": float(local_always.mean()),
            "boot_share_decisions_cheaper_at_all_c": float(hosted_always.mean()),
            "boot_share_other": float(other.mean()),
            "boot_full_p2.5 (inf=local always cheaper)": float(np.percentile(full[~other], 2.5, method="nearest")),
            "boot_full_median": float(np.percentile(full[~other], 50, method="nearest")),
            "boot_full_p97.5": float(np.percentile(full[~other], 97.5, method="nearest"))}


def envelope_bootstrap(Ml: np.ndarray, Fl: np.ndarray, Md, Fd, n_boot: int) -> dict[str, Any]:
    """Best-local envelope: the largest c below which SOME local model is cheaper. Ml/Fl: (n_boot, n_local)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        csl = np.where(Ml > Md[:, None], (Fd[:, None] - Fl) / (Ml - Md[:, None]), np.nan)
    never_worse = (Ml <= Md[:, None]) & (Fl <= Fd[:, None])
    env = np.nanmax(np.where(np.isnan(csl), -np.inf, csl), axis=1)
    env_inf = never_worse.any(1)
    envd = env[(~env_inf) & np.isfinite(env) & (env > 0)]
    envfull = np.where(env_inf, np.inf, np.where(np.isfinite(env) & (env > 0), env, 0.0))
    return {"boot_share_local_cheaper_below_cstar": float(len(envd) / n_boot),
            "boot_share_some_local_dominates_decisions": float(env_inf.mean()),
            "boot_c_star_p2.5": float(np.percentile(envd, 2.5)) if envd.size else None,
            "boot_c_star_median": float(np.median(envd)) if envd.size else None,
            "boot_c_star_p97.5": float(np.percentile(envd, 97.5)) if envd.size else None,
            "boot_full_p2.5 (inf=local always cheaper)": float(np.percentile(envfull, 2.5, method="nearest")),
            "boot_full_median": float(np.percentile(envfull, 50, method="nearest")),
            "boot_full_p97.5": float(np.percentile(envfull, 97.5, method="nearest"))}


def min_cost_probability(MISS: dict, FE: dict, kind: str, pool: list[str], c: float) -> np.ndarray:
    """Share of replicates in which each pool member has the lowest c*misses+FE (ties split)."""
    cost = np.stack([c * MISS[(kind, m)].sum(1) + FE[(kind, m)].sum(1) for m in pool], 1)
    win = (cost == cost.min(1, keepdims=True)).astype(float)
    win /= win.sum(1, keepdims=True)
    return win.mean(0)


# --------------------------------------------------------------------------- paired tests


def paired(a, b, seed: int = 42, n_boot: int = 10000) -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank and exact sign test over units (organizations or tenants)."""
    from scipy.stats import binomtest, wilcoxon

    d = np.array(a) - np.array(b)
    d = d[~np.isnan(d)]
    nz = d[d != 0]
    res = {"n_orgs": int(len(d)), "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
           "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()), "n_tie": int((d == 0).sum())}
    res["wilcoxon_p"] = float(wilcoxon(d, zero_method="wilcox").pvalue) if len(nz) else float("nan")
    res["sign_p"] = float(binomtest(res["n_pos"], len(nz), 0.5).pvalue) if len(nz) else float("nan")
    if len(nz):
        ranks = np.argsort(np.argsort(np.abs(nz))) + 1.0
        absd = np.abs(nz)
        for v in np.unique(absd):
            ranks[absd == v] = ranks[absd == v].mean()
        res["rank_biserial"] = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, len(d)), replace=True)
    res["mean_ci95"] = [float(x) for x in np.percentile(boots.mean(axis=1), [2.5, 97.5])]
    res["median_ci95"] = [float(x) for x in np.percentile(np.median(boots, axis=1), [2.5, 97.5])]
    return res


def holm(ps: Sequence[float]) -> list[float]:
    order = np.argsort(ps)
    m, adj, running = len(ps), [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * ps[i]))
        adj[i] = running
    return adj


def mcnemar_exact(y: Sequence[str], pred_a: Sequence[str], pred_b: Sequence[str], positive=MAL) -> float:
    """Exact McNemar test on per-item correctness of the malicious-vs-rest decision (used per tenant)."""
    from scipy.stats import binomtest

    ca = [ism(p) == ism(t) for p, t in zip(pred_a, y)]
    cb = [ism(p) == ism(t) for p, t in zip(pred_b, y)]
    b = sum(x and not z for x, z in zip(ca, cb))
    c = sum(z and not x for x, z in zip(ca, cb))
    return float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0


# --------------------------------------------------------------------------- prevalence reweighting


def reweight_factor(n_mal_target: int, n_nonmal_target: int, n_mal: int, n_nonmal: int) -> float:
    """Weight w for each non-malicious item so the weighted class ratio equals the target ratio."""
    return (n_nonmal_target / n_mal_target) / (n_nonmal / n_mal)


def weighted_gate(recs: list[dict], w: float, coverage: float = 0.8) -> tuple[float, float]:
    """Test-quantile gate on weighted mass: keep the most confident items until >= coverage of the mass."""
    rk = ranked(recs)
    tot = sum(1.0 if ism(r["y"]) else w for r in rk)
    acc, miss, fe = 0.0, 0, 0.0
    for r in rk:
        if acc >= coverage * tot - 1e-9:
            break
        acc += 1.0 if ism(r["y"]) else w
        if ism(r["y"]) and not ism(r["pred"]):
            miss += 1
        if (not ism(r["y"])) and ism(r["pred"]):
            fe += w
    return miss, fe
