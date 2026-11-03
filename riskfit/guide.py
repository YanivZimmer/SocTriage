"""Public GUIDE replication: incident aggregation, distinct-prompt collapse, organization selection, splits.

Implements the pre-registered rule in ``configs/guide_selection.yaml``:

* GUIDE_Train.csv and GUIDE_Test.csv are pooled; an incident is keyed by (OrgId, IncidentId) and its
  evidence rows are aggregated into one normalized alert record (``incident_record``). Incidents with no
  grade or more than one distinct grade are dropped. Grades map TruePositive -> malicious,
  BenignPositive -> benign, FalsePositive -> false positive.
* Organizations 4 and 9 are excluded; eligible organizations have >= 300 incidents, >= 10 malicious and
  >= 10 of each other grade, and are visited in ascending sha256("X4-GUIDE-org-" + OrgId) order.
* The unit of prediction is a distinct prompt: incidents whose rendered text is byte-identical collapse to
  one prompt labelled by the incidents' majority grade (ties: malicious > benign > false positive). An
  organization is kept if it has >= 300 prompts and >= 10 malicious-majority prompts and its 70/15/15
  stratified split (seed 42) puts >= 1 malicious prompt in validation and in test. The first 12 kept
  organizations form the benchmark.
* Each split file is deterministic JSONL; the split fingerprint (sha256 over the three files' record
  counts and sha256) must equal ``configs/guide_orgs.csv:split_fingerprint``.

The hash salt "X4-GUIDE-org-" is part of the pre-registered rule and must not be changed.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .serialization import render_alert

FILES = ("GUIDE_Train.csv", "GUIDE_Test.csv")
GUIDE_LABEL_MAP = {
    "TruePositive": "True Positive - Malicious",
    "BenignPositive": "True Positive - Benign",
    "FalsePositive": "False Positive",
}
MAL = "True Positive - Malicious"
TIE_ORDER = [MAL, "True Positive - Benign", "False Positive"]
SOURCE_COLUMNS = ("OrgId", "IncidentId", "Timestamp", "Category", "MitreTechniques", "EntityType",
                  "EvidenceRole", "IncidentGrade")
ORDER_SALT = "X4-GUIDE-org-"


# --------------------------------------------------------------------------- hashing helpers


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def render_jsonl(records: list[dict[str, Any]]) -> bytes:
    lines = [json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False) for r in records]
    return ("\n".join(lines) + "\n").encode("utf-8")


def split_metadata(filename: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "filename": filename,
        "record_count": len(records),
        "sha256": hashlib.sha256(render_jsonl(records)).hexdigest(),
        "incident_id_fingerprint": sha256_json(sorted(r["alert_id"] for r in records)),
        "label_counts": dict(sorted(Counter(r["label"] for r in records).items())),
    }


def split_fingerprint(meta: dict[str, dict[str, Any]]) -> str:
    return sha256_json({k: {"record_count": m["record_count"], "sha256": m["sha256"]} for k, m in meta.items()})


def org_order_key(org: str) -> str:
    return hashlib.sha256(f"{ORDER_SALT}{org}".encode()).hexdigest()


# --------------------------------------------------------------------------- incidents


def _top(values: Counter, limit: int = 25) -> list[str]:
    return [v for v, _ in sorted(values.items(), key=lambda it: (-it[1], it[0]))[:limit]]


def _add(counter: Counter, value: object) -> None:
    import pandas as pd

    if pd.isna(value):
        return
    s = str(value).strip()
    if s:
        counter[s] += 1


def _empty() -> dict[str, Any]:
    return {"timestamps": [], "labels": Counter(), "categories": Counter(), "mitre_techniques": Counter(),
            "entity_types": Counter(), "evidence_roles": Counter(), "evidence_count": 0}


def incident_record(org: str, incident_id: str, agg: dict[str, Any]) -> dict[str, Any]:
    """One GUIDE incident as a normalized alert record (identifiers excluded from the rendered text)."""
    (grade,) = agg["labels"]
    cats, mitre = _top(agg["categories"]), _top(agg["mitre_techniques"])
    ents, roles = _top(agg["entity_types"]), _top(agg["evidence_roles"])
    summary = " | ".join((
        f"evidence_count={agg['evidence_count']}",
        f"categories={','.join(cats) or 'missing'}",
        f"mitre_techniques={','.join(mitre) or 'missing'}",
        f"entity_types={','.join(ents) or 'missing'}",
        f"evidence_roles={','.join(roles) or 'missing'}",
    ))
    observables = ([{"type": "guide_category", "value": v, "reputation": "unknown"} for v in cats]
                   + [{"type": "guide_entity_type", "value": v, "reputation": "unknown"} for v in ents])
    if not observables:
        observables = [{"type": "guide_evidence", "value": "present", "reputation": "unknown"}]
    return {
        "id": f"guide-{org}-{incident_id}",
        "alert_id": f"guide-{org}-{incident_id}",
        "tenant_id": f"guide-org-{org}",
        "source": "Microsoft GUIDE",
        "name": cats[0] if cats else "GUIDE incident",
        "category": cats[0] if cats else None,
        "severity": "medium",
        "detected_at": min(agg["timestamps"]) if agg["timestamps"] else None,
        "detection": {"description": summary, "risk_score": float(agg["evidence_count"])},
        "mitre_attack_techniques": [{"code": v, "name": v} for v in mitre],
        "observables": observables,
        "label": GUIDE_LABEL_MAP[grade],
        "label_source": "guide_analyst",
    }


def aggregate_incidents(source_dir: Path, orgs: set[str], chunksize: int = 500_000,
                        nrows: int | None = None) -> dict[str, list[dict[str, Any]]]:
    """Aggregate evidence rows of both GUIDE files into labelled incident records for ``orgs``."""
    import pandas as pd

    aggs: dict[tuple[str, str], dict[str, Any]] = defaultdict(_empty)
    for name in FILES:
        for chunk in pd.read_csv(Path(source_dir) / name, usecols=SOURCE_COLUMNS, chunksize=chunksize,
                                 low_memory=False, nrows=nrows):
            sel = chunk[chunk["OrgId"].astype(str).isin(orgs)]
            for row in sel.itertuples(index=False):
                a = aggs[(str(row.OrgId), str(row.IncidentId))]
                _add(a["labels"], row.IncidentGrade)
                _add(a["categories"], row.Category)
                _add(a["mitre_techniques"], row.MitreTechniques)
                _add(a["entity_types"], row.EntityType)
                _add(a["evidence_roles"], row.EvidenceRole)
                if not pd.isna(row.Timestamp):
                    a["timestamps"].append(str(row.Timestamp))
                a["evidence_count"] += 1
    grouped: dict[str, list[dict[str, Any]]] = {o: [] for o in orgs}
    for (org, inc), a in sorted(aggs.items()):
        if len(a["labels"]) != 1:  # no grade or conflicting grades
            continue
        grouped[org].append(incident_record(org, inc, a))
    return grouped


def org_label_counts(source_dir: Path, chunksize: int = 1_000_000):
    """Pass 1: per-organization counts of consistently graded incidents (both files pooled)."""
    import pandas as pd

    parts = []
    for name in FILES:
        for chunk in pd.read_csv(Path(source_dir) / name, usecols=["OrgId", "IncidentId", "IncidentGrade"],
                                 chunksize=chunksize, low_memory=False):
            parts.append(chunk.drop_duplicates().dropna(subset=["IncidentGrade"]))
    grades = pd.concat(parts, ignore_index=True).drop_duplicates()
    n_grades = grades.groupby(["OrgId", "IncidentId"])["IncidentGrade"].nunique()
    consistent = n_grades[n_grades == 1].index
    labeled = grades.set_index(["OrgId", "IncidentId"]).loc[consistent].reset_index()
    counts = labeled.groupby(["OrgId", "IncidentGrade"]).size().unstack(fill_value=0)
    for g in GUIDE_LABEL_MAP:
        if g not in counts:
            counts[g] = 0
    counts["n"] = counts[list(GUIDE_LABEL_MAP)].sum(axis=1)
    return counts


def candidate_orgs(counts, rule: dict[str, Any]) -> list[str]:
    """Incident-level eligibility, in the pre-registered hash order (no upper size cap)."""
    out = [str(int(o)) for o, r in counts.iterrows()
           if str(int(o)) not in set(rule["exclude_orgs"]) and r["n"] >= rule["min_incidents"]
           and r["TruePositive"] >= rule["min_malicious"]
           and min(r["BenignPositive"], r["FalsePositive"]) >= rule["min_each_other_class"]]
    return sorted(out, key=org_order_key)


# --------------------------------------------------------------------------- prompts and splits


def collapse(org: str, recs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collapse incidents with byte-identical rendered text into distinct prompts."""
    texts = [render_alert(r) for r in recs]
    groups: dict[str, list[int]] = {}
    for i, t in enumerate(texts):
        groups.setdefault(t, []).append(i)
    prompts, ties = [], []
    for text, idx in groups.items():
        idx = sorted(idx, key=lambda i: recs[i]["alert_id"])
        counts = Counter(recs[i]["label"] for i in idx)
        best = max(counts.values())
        tied = [lab for lab in TIE_ORDER if counts.get(lab, 0) == best]
        pid = f"guide-{org}-p{hashlib.sha256(text.encode()).hexdigest()[:16]}"
        if len(tied) > 1:
            ties.append({"org_id": org, "prompt_id": pid, "label_counts": dict(counts), "broken_to": tied[0]})
        rep = copy.deepcopy(recs[idx[0]])
        rep["id"] = rep["alert_id"] = pid
        rep["label"] = tied[0]
        prompts.append({"record": rep, "text": text, "label": tied[0], "labels": counts,
                        "incidents": [recs[i] for i in idx]})
    assert all(render_alert(p["record"]) == p["text"] for p in prompts), "representative must re-render identically"
    return sorted(prompts, key=lambda p: p["record"]["alert_id"]), ties


def split_prompts(prompts: list[dict[str, Any]], seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    from sklearn.model_selection import train_test_split

    rest, test = train_test_split(prompts, test_size=0.15, random_state=seed, stratify=[p["label"] for p in prompts])
    train, val = train_test_split(rest, test_size=0.15 / 0.85, random_state=seed, stratify=[p["label"] for p in rest])
    return {"train": train, "val": val, "test": test}


def build_org(org: str, recs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Collapse, check prompt-level eligibility and feasibility, split, and fingerprint one organization."""
    prompts, ties = collapse(org, recs)
    n_mal = sum(p["label"] == MAL for p in prompts)
    if len(prompts) < 300 or n_mal < 10:
        return {"skip": f"ineligible: {len(prompts)} prompts, {n_mal} malicious-majority"}
    parts = split_prompts(prompts)
    nmp = {k: sum(p["label"] == MAL for p in v) for k, v in parts.items()}
    if nmp["val"] < 1 or nmp["test"] < 1:
        return {"skip": f"infeasible split {nmp}"}
    split_recs = {k: sorted((p["record"] for p in v), key=lambda r: r["alert_id"]) for k, v in parts.items()}
    meta = {k: split_metadata(f"{k}.jsonl", split_recs[k]) for k in ("train", "val", "test")}
    split_of = {p["record"]["alert_id"]: k for k, v in parts.items() for p in v}
    membership = [[org, inc["alert_id"].split("-")[-1], p["record"]["alert_id"], split_of[p["record"]["alert_id"]],
                   inc["label"], p["label"]] for p in prompts for inc in p["incidents"]]

    def noise(ps):
        pairs = [(i["label"], p["label"]) for p in ps for i in p["incidents"]]
        return sum(a != b for a, b in pairs) / len(pairs), len(pairs), sum(a == MAL for a, _ in pairs)

    row = {"org_id": org, "n": len(recs), "n_mal": sum(r["label"] == MAL for r in recs), "split_seed": 42,
           "n_prompts": len(prompts), "n_mal_prompts": n_mal,
           "n_benign_prompts": sum(p["label"] == "True Positive - Benign" for p in prompts),
           "n_fp_prompts": sum(p["label"] == "False Positive" for p in prompts),
           "mixed_label_prompts": sum(len(p["labels"]) > 1 for p in prompts), "tied_prompts": len(ties),
           "noise_share_org": round(noise(prompts)[0], 4)}
    for k, v in parts.items():
        share, n_inc, n_inc_mal = noise(v)
        row.update({f"n_prompts_{k}": len(v), f"n_mal_prompts_{k}": nmp[k], f"n_incidents_{k}": n_inc,
                    f"n_mal_incidents_{k}": n_inc_mal, f"noise_share_{k}": round(share, 4)})
    row.update({"unit": "distinct prompt (majority label)", "split": "stratified 70/15/15 over prompts, seed 42",
                "split_fingerprint": split_fingerprint(meta)})
    return {"row": row, "split_records": split_recs, "meta": meta, "membership": membership, "ties": ties}


def write_org(out_dir: Path, org: str, built: dict[str, Any], archive_sha256: str | None) -> Path:
    d = Path(out_dir) / "splits" / f"org_{org}"
    d.mkdir(parents=True, exist_ok=True)
    for k in ("train", "val", "test"):
        (d / f"{k}.jsonl").write_bytes(render_jsonl(built["split_records"][k]))
    manifest = {"version": 3, "dataset": "GUIDE", "unit_of_prediction": "distinct_prompt", "tenant": org,
                "source_files": list(FILES), "archive_sha256": archive_sha256, "split_seed": 42,
                "test_ratio": 0.15, "validation_ratio": 0.15, "label_mapping": GUIDE_LABEL_MAP,
                "prompt_label": "majority incident label; ties broken malicious > benign > FP",
                "splits": built["meta"], "fingerprint": built["row"]["split_fingerprint"]}
    (d / "benchmark_split_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return d


# --------------------------------------------------------------------------- loading built splits


def load_split(splits_dir: Path, org: str) -> dict[str, list[dict[str, Any]]]:
    """Load one organization's split files and verify them against its manifest (hashes, disjointness)."""
    d = Path(splits_dir) / f"org_{org}"
    man = json.loads((d / "benchmark_split_manifest.json").read_text())
    out = {}
    for k in ("train", "val", "test"):
        b = (d / f"{k}.jsonl").read_bytes()
        assert hashlib.sha256(b).hexdigest() == man["splits"][k]["sha256"], f"org {org} {k}: hash mismatch"
        out[k] = [json.loads(line) for line in b.decode().splitlines() if line.strip()]
    ids = {k: {r["alert_id"] for r in v} for k, v in out.items()}
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"]), "split overlap"
    return out


def iter_orgs(orgs_csv: Path) -> Iterable[dict[str, str]]:
    import csv

    with Path(orgs_csv).open(newline="") as fh:
        yield from csv.DictReader(fh)
