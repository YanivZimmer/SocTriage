#!/usr/bin/env python3
"""Build the 12-organization GUIDE benchmark (distinct-prompt unit) and verify it against the paper's splits.

Two modes:
  default   rebuild the organizations listed in configs/guide_orgs.csv (or --orgs) and check each split
            fingerprint; with all 12, also check the incident->prompt->split membership and tie tables.
  --select  re-run the whole pre-registered selection walk (pass 1 counts, hash order, eligibility and
            feasibility checks) and check that it selects the same 12 organizations in the same order.

Outputs (in --out): splits/org_<id>/{train,val,test}.jsonl + benchmark_split_manifest.json,
guide_orgs.csv, guide_prompt_membership.csv, guide_prompt_ties.csv, selection_audit.json.
Public GUIDE data only; nothing leaves the machine.

Usage:
  python scripts/build_guide_splits.py --source-dir data/guide_raw --out data/guide_splits [--archive data/guide_raw/archive.zip]
  python scripts/build_guide_splits.py --source-dir data/guide_raw --out data/guide_splits --orgs 78
  python scripts/build_guide_splits.py --source-dir data/guide_raw --out data/guide_splits_select --select
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from riskfit import guide as G  # noqa: E402

CFG = ROOT / "configs"


def csv_bytes(header, rows, dict_rows=False) -> bytes:
    buf = io.StringIO(newline="")
    if dict_rows:
        w = csv.DictWriter(buf, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    else:
        w = csv.writer(buf)
        w.writerow(header)
        w.writerows(rows)
    return buf.getvalue().encode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, required=True, help="directory with GUIDE_Train.csv and GUIDE_Test.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--archive", type=Path, help="archive.zip to check against the pre-registered sha256")
    ap.add_argument("--orgs", help="comma-separated subset of the 12 organizations (default: all)")
    ap.add_argument("--select", action="store_true", help="re-run the full pre-registered selection walk")
    ap.add_argument("--batch", type=int, default=25, help="organizations aggregated per pass in --select mode")
    a = ap.parse_args()

    rule = yaml.safe_load((CFG / "guide_selection.yaml").read_text())
    expected = rule["expected"]
    want = {r["org_id"]: r for r in G.iter_orgs(CFG / "guide_orgs.csv")}
    archive_sha = rule["archive"]["archive_sha256"]
    if a.archive:
        got = G.sha256_file(a.archive)
        print("archive sha256", "OK" if got == archive_sha else f"MISMATCH {got}", flush=True)
        if got != archive_sha:
            return 1
    a.out.mkdir(parents=True, exist_ok=True)

    selected, skipped, rows, ties, membership = [], [], [], [], []
    if a.select:
        counts = G.org_label_counts(a.source_dir)
        order = G.candidate_orgs(counts, rule["selection"])
        print(f"{len(order)} incident-eligible organizations in hash order", flush=True)
        pos = 0
        while len(selected) < rule["selection"]["take"] and pos < len(order):
            chunk = order[pos:pos + a.batch]
            grouped = G.aggregate_incidents(a.source_dir, set(chunk))
            for org in chunk:
                if len(selected) >= rule["selection"]["take"]:
                    break
                pos += 1
                built = G.build_org(org, grouped[org])
                if "skip" in built:
                    skipped.append({"org_id": org, "hash_rank": pos, "reason": built["skip"]})
                    continue
                selected.append(org)
                built["row"]["hash_rank"] = pos
                G.write_org(a.out, org, built, archive_sha)
                rows.append(built["row"]); ties += built["ties"]; membership += built["membership"]
                print(f"selected org {org} (hash rank {pos})", flush=True)
        orgs = selected
        ok = selected == expected["orgs_in_order"] and pos == expected["orgs_visited_in_hash_order"]
        print("selection", "MATCHES the paper" if ok else f"DIFFERS: {selected} after {pos} visits", flush=True)
    else:
        orgs = a.orgs.split(",") if a.orgs else list(want)
        unknown = [o for o in orgs if o not in want]
        assert not unknown, f"not one of the 12 benchmark organizations: {unknown}"
        grouped = G.aggregate_incidents(a.source_dir, set(orgs))
        ok = True
        for org in orgs:
            built = G.build_org(org, grouped[org])
            assert "skip" not in built, f"org {org}: {built.get('skip')}"
            built["row"]["hash_rank"] = int(want[org]["hash_rank"])
            G.write_org(a.out, org, built, archive_sha)
            rows.append(built["row"]); ties += built["ties"]; membership += built["membership"]
            good = built["row"]["split_fingerprint"] == want[org]["split_fingerprint"]
            ok &= good
            n = built["row"]
            print(f"org {org}: prompts train/val/test = {n['n_prompts_train']}/{n['n_prompts_val']}/{n['n_prompts_test']}, "
                  f"fingerprint {'OK' if good else 'MISMATCH'}", flush=True)

    (a.out / "guide_orgs.csv").write_bytes(csv_bytes(list(rows[0]), rows, dict_rows=True))
    mem = csv_bytes(["org_id", "incident_id", "prompt_id", "split", "incident_label", "prompt_label"], membership)
    (a.out / "guide_prompt_membership.csv").write_bytes(mem)
    tie_rows = [{**t, "label_counts": json.dumps(t["label_counts"], sort_keys=True)} for t in ties]
    tie_b = csv_bytes(["org_id", "prompt_id", "label_counts", "broken_to"], tie_rows, dict_rows=True)
    (a.out / "guide_prompt_ties.csv").write_bytes(tie_b)
    (a.out / "selection_audit.json").write_text(json.dumps(
        {"mode": "select" if a.select else "listed", "selected": orgs, "skipped": skipped}, indent=2) + "\n")
    if sorted(orgs) == sorted(expected["orgs_in_order"]):
        for name, b, key in (("membership", mem, "membership_csv_sha256"), ("ties", tie_b, "ties_csv_sha256")):
            good = hashlib.sha256(b).hexdigest() == expected[key]
            ok &= good
            print(f"{name} table sha256 {'OK' if good else 'MISMATCH'}", flush=True)
        n_test = sum(r["n_prompts_test"] for r in rows)
        n_mal = sum(r["n_mal_prompts_test"] for r in rows)
        print(f"test prompts {n_test} ({n_mal} malicious); expected {expected['test_prompts']} ({expected['test_malicious_prompts']})")
        ok &= (n_test, n_mal) == (expected["test_prompts"], expected["test_malicious_prompts"])
    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
