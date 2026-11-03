#!/usr/bin/env python3
"""Build the hosted-model inputs (zero-shot and one-exemplar user messages) for the GUIDE prompts.

No model or API is called. Checks: the instruction texts hash to the values used in the paper's runs, every
prompt id re-derives from its rendered text, every exemplar comes from the same organization's training
split, and (when data/guide_outputs/ is present) every exemplar id equals the one in the released outputs.

Usage:
  python scripts/build_hosted_inputs.py --splits data/guide_splits/splits --out data/hosted_inputs.jsonl
  python scripts/build_hosted_inputs.py ... --orgs 78 --splits-to-score test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from riskfit import guide as G  # noqa: E402
from riskfit import prompts as P  # noqa: E402

# Exemplar diagnostics recorded for the full 12-organization run (test + validation).
EXPECTED_DIAG = {
    "test_exemplar_label_benign": 531, "test_exemplar_label_false_positive": 63, "test_exemplar_label_malicious": 485,
    "test_exemplar_same_label_as_target": 929, "test_matched": 1075, "test_no_match": 4,
    "val_exemplar_label_benign": 544, "val_exemplar_label_false_positive": 60, "val_exemplar_label_malicious": 475,
    "val_exemplar_same_label_as_target": 927, "val_matched": 1076, "val_no_match": 3,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--orgs", help="comma-separated organizations (default: all 12)")
    ap.add_argument("--splits-to-score", default="test,val")
    ap.add_argument("--k", type=int, default=1, help="exemplars per prompt (1 = the paper; 5 = the follow-up)")
    ap.add_argument("--released", type=Path, default=ROOT / "data" / "guide_outputs" / "decisions_guide_per_prompt.csv")
    a = ap.parse_args()

    ok = True
    for k, good in P.check_instruction_hashes().items():
        print(f"[{'PASS' if good else 'FAIL'}] {k} sha256 matches the paper's runs")
        ok &= good
    all_orgs = [r["org_id"] for r in G.iter_orgs(ROOT / "configs" / "guide_orgs.csv")]
    orgs = a.orgs.split(",") if a.orgs else all_orgs
    which = a.splits_to_score.split(",")
    splits = {o: G.load_split(a.splits, o) for o in orgs}
    items = [it for o in sorted(orgs, key=int) for it in P.build_org_inputs(o, splits[o], which=which, k=a.k)]
    assert all(i["ex_user"] is not None for i in items), "some target has no exemplar"
    diag = P.exemplar_diagnostics(items, splits)
    print("exemplar diagnostics:", json.dumps(diag, sort_keys=True))
    if sorted(orgs) == sorted(all_orgs) and set(which) == {"test", "val"}:
        good = diag == EXPECTED_DIAG
        print(f"[{'PASS' if good else 'FAIL'}] exemplar diagnostics equal the paper's run")
        ok &= good
    if a.released.exists():
        rel = {(r["split"], r["prompt_id"]): r for r in csv.DictReader(a.released.open(newline=""))}
        bad = sum(rel[(i["split"], i["prompt_id"])]["exemplar_prompt_id"] != i["exemplar_prompt_id"]
                  or rel[(i["split"], i["prompt_id"])]["label"] != i["label"] for i in items)
        print(f"[{'PASS' if not bad else 'FAIL'}] exemplar ids and labels equal the released outputs ({len(items)} prompts, {bad} differ)")
        ok &= bad == 0
    k5 = ROOT / "data" / "guide_outputs" / "decisions_jev_5ex_guide_per_prompt.csv"
    if a.k == 5 and k5.exists():
        rel5 = {(r["split"], r["prompt_id"]): r for r in csv.DictReader(k5.open(newline=""))}
        bad = sum(rel5[(i["split"], i["prompt_id"])]["user5_sha256"] != P.sha256_text(i["ex_user"])
                  or [rel5[(i["split"], i["prompt_id"])][f"ex{j}_prompt_id"] for j in range(1, 6)] != i["exemplar_prompt_ids"]
                  for i in items)
        print(f"[{'PASS' if not bad else 'FAIL'}] five-exemplar messages equal the released follow-up ({bad} differ)")
        ok &= bad == 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as fh:
        for i in items:
            fh.write(json.dumps(i, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} items to {a.out}")
    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
