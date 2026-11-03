#!/usr/bin/env python3
"""OpenAI Decisions API (POST https://api.openai.com/v1/decisions, model gpt-6-luna) on the GUIDE prompts.

Each prompt is sent zero-shot and with its exemplar; the request asks two questions (riskfit.prompts):
q1 a choice over [false_positive, benign, malicious] (verdict, confidence, per-choice probabilities) and q2 a
predicate "the target alert is malicious" (probability). The paper's run (2026-11-01, 4,316 calls) returned
the model id gpt-6-luna (no dated snapshot) and was deterministic across repeated calls.

The key is read from the OPENAI_API_KEY environment variable only. Every response is cached under --cache.
Writes the released layout (data/guide_outputs/decisions_guide_per_prompt.csv).

Usage: OPENAI_API_KEY=... python scripts/run_decisions.py --inputs data/hosted_inputs.jsonl --out out/decisions_guide_per_prompt.csv
       [--limit 4] [--workers 12]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from riskfit import hosted as H  # noqa: E402
from riskfit.prompts import VERDICTS, decisions_questions  # noqa: E402

URL = "https://api.openai.com/v1/decisions"
MODEL = "gpt-6-luna"


class Transient(Exception):
    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("cache/decisions"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    import httpx

    questions = decisions_questions()
    client = httpx.Client(timeout=120, headers={"Authorization": "Bearer " + H.require_env("OPENAI_API_KEY")})

    def call(item, cond):
        def once():
            r = client.post(URL, json={"model": MODEL, "input": item[f"{cond}_user"], "questions": questions})
            if r.status_code == 429 or r.status_code >= 500:
                raise Transient(str(r.status_code))
            if r.status_code != 200:
                return {"verdict": None, "error": f"http{r.status_code}"}
            d = r.json()
            ans = {x.get("name"): x for x in d.get("answers", [])}
            v, p = ans.get("verdict", {}), ans.get("is_malicious", {})
            if v.get("type") != "choice" or p.get("type") != "predicate":
                return {"verdict": None, "error": "refusal_or_missing"}
            probs = {str(x["value"]): float(x["probability"]) for x in v.get("probabilities", [])}
            return {"verdict": str(v["choice"]), "confidence": float(v["confidence"]),
                    "probs": [probs.get(lab, 0.0) for lab in VERDICTS], "p_q2": float(p["probability"]),
                    "model": d.get("model")}
        return H.with_retries(once, retry_on=lambda e: isinstance(e, (Transient, httpx.TransportError)))

    items = H.load_inputs(a.inputs, limit=a.limit)
    res = H.run_all([(i, c) for i in items for c in ("ex", "zs")], call, H.Cache(a.cache),
                    {"model": MODEL, "questions": questions}, a.workers)
    rows = []
    for i in items:
        row = {k: i[k] for k in ("org", "split", "prompt_id", "label", "exemplar_prompt_id", "exemplar_status",
                                 "exemplar_score")}
        for c in ("ex", "zs"):
            r = res[(i["split"], i["prompt_id"], c)]
            pr = r.get("probs") or [None] * 3
            row.update({f"{c}_verdict": r.get("verdict"), f"{c}_confidence": r.get("confidence"), f"{c}_p_fp": pr[0],
                        f"{c}_p_benign": pr[1], f"{c}_p_mal_q1": pr[2], f"{c}_p_mal_q2": r.get("p_q2")})
            if c == "ex":
                row["ex_error"] = r.get("error")
        rows.append(row)
    H.write_rows(a.out, rows)
    print(f"wrote {len(rows)} rows to {a.out}; unresolved:", sum(r[f'{c}_verdict'] in (None, '') for r in rows for c in ('ex', 'zs')))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
