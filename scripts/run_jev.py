#!/usr/bin/env python3
"""TypeSafe Jev (POST https://api.typesafe.ai/v1/systemone, model jev-latest) on the GUIDE prompts.

Byte-identical user messages to the Decisions run (sent as `state`); the same choice question (choice
descriptions as `criteria`) and the predicate as Jev's yes/no ('noul') question, which returns P(yes). The
paper's run (2026-11-01) echoed model jev-1.14.0 and cost USD 0.15.

The key is read from the TYPESAFE_API_KEY environment variable only. Every response is cached under --cache.
Writes the released layout (data/guide_outputs/jev_guide_per_prompt.csv).

Usage: TYPESAFE_API_KEY=... python scripts/run_jev.py --inputs data/hosted_inputs.jsonl --out out/jev_guide_per_prompt.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from riskfit import hosted as H  # noqa: E402
from riskfit.prompts import VERDICTS, jev_questions  # noqa: E402

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"


class Transient(Exception):
    pass


def parse(d: dict) -> dict:
    ans = d.get("answers") or {}
    v, p = ans.get("verdict") or {}, ans.get("is_malicious") or {}
    if v.get("type") != "choice" or p.get("type") != "noul":
        return {"verdict": None, "error": "missing_or_wrong_type"}
    probs = v.get("probabilities") or {}
    if isinstance(probs, list):
        probs = {str(x.get("value", x.get("choice"))): x.get("probability") for x in probs}
    return {"verdict": str(v["choice"]), "confidence": float(v["confidence"]),
            "probs": [float(probs.get(lab, 0.0)) for lab in VERDICTS], "p_q2": float(p["noul"]),
            "q2_confidence": float(p["confidence"]) if p.get("confidence") is not None else None,
            "input_tokens": (d.get("usage") or {}).get("input_tokens"), "model": d.get("model")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("cache/jev"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    import httpx

    questions = jev_questions()
    client = httpx.Client(timeout=60, headers={"Authorization": "Bearer " + H.require_env("TYPESAFE_API_KEY"),
                                               "Content-Type": "application/json"})

    def call(item, cond):
        def once():
            r = client.post(URL, json={"state": item[f"{cond}_user"], "model": MODEL, "questions": questions})
            if r.status_code == 429 or r.status_code >= 500:
                raise Transient(str(r.status_code))
            if r.status_code != 200:
                return {"verdict": None, "error": f"http{r.status_code}"}
            return parse(r.json())
        return H.with_retries(once, retry_on=lambda e: isinstance(e, (Transient, httpx.TransportError)))

    items = H.load_inputs(a.inputs, limit=a.limit)
    res = H.run_all([(i, c) for i in items for c in ("zs", "ex")], call, H.Cache(a.cache),
                    {"model": MODEL, "questions": questions}, a.workers)
    rows = []
    for i in items:
        row = {k: i[k] for k in ("org", "split", "prompt_id", "label", "exemplar_prompt_id", "exemplar_status",
                                 "exemplar_score")}
        for c in ("zs", "ex"):
            r = res[(i["split"], i["prompt_id"], c)]
            pr = r.get("probs") or [None] * 3
            row.update({f"{c}_verdict": r.get("verdict"), f"{c}_confidence": r.get("confidence"), f"{c}_p_fp": pr[0],
                        f"{c}_p_benign": pr[1], f"{c}_p_mal_q1": pr[2], f"{c}_p_mal_q2": r.get("p_q2"),
                        f"{c}_q2_confidence": r.get("q2_confidence"), f"{c}_error": r.get("error"),
                        f"{c}_input_tokens": r.get("input_tokens")})
        rows.append(row)
    H.write_rows(a.out, rows)
    print(f"wrote {len(rows)} rows to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
