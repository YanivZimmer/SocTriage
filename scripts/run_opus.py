#!/usr/bin/env python3
"""Claude Opus 5.5 (claude-opus-5-5, effort medium, no tools) on the GUIDE test prompts.

User messages are byte-identical to the decision models' inputs; the system prompt carries the same choice
and predicate instructions plus a JSON output format {verdict, confidence, p_malicious}
(riskfit.prompts.opus_system_prompt). One call per prompt and condition (zero-shot, with exemplar).

Two transports:
  --via api   (default) Anthropic Messages API through the official SDK; key from ANTHROPIC_API_KEY
              (or any credential the SDK resolves). Adaptive thinking is always on for this model; effort
              is set explicitly to medium; sampling parameters are not accepted by the model.
              Server-side refusal fallbacks are deliberately NOT enabled, so every answer comes from
              claude-opus-5-5; a refusal is recorded as unresolved.
  --via cli   the path used for the paper's run (2026-11-03): the Claude Code CLI in print mode, in an empty
              working directory with tools, hooks, MCP servers, slash commands and session persistence off:
                claude -p --model claude-opus-5-5 --effort medium --tools "" --strict-mcp-config
                       --disable-slash-commands --settings '{"disableAllHooks":true}'
                       --no-session-persistence --system-prompt <system> --output-format json
              The CLI uses whatever account it is logged in to; set CLAUDE_CLI to its path if not on PATH.
The paper's run was a single, non-deterministic pass; expect small differences on a rerun.

Usage: python scripts/run_opus.py --inputs data/hosted_inputs.jsonl --out out/opus_guide_per_prompt.csv [--via cli] [--limit 4]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from riskfit import hosted as H  # noqa: E402
from riskfit.prompts import VERDICTS, opus_system_prompt  # noqa: E402

MODEL, EFFORT = "claude-opus-5-5", "medium"


def parse(text: str | None) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {"verdict": None, "error": "no_json"}
    try:
        d = json.loads(m.group(0))
        v, c, p = d["verdict"], float(d["confidence"]), float(d["p_malicious"])
    except (ValueError, KeyError, TypeError):
        return {"verdict": None, "error": "bad_json"}
    if v not in VERDICTS or not (0 <= c <= 1 and 0 <= p <= 1):
        return {"verdict": None, "error": "out_of_range"}
    return {"verdict": v, "confidence": c, "p_malicious": p}


def api_caller(system: str):
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    def call(item, cond):
        def once():
            resp = client.messages.create(model=MODEL, max_tokens=16000, system=system,
                                          output_config={"effort": EFFORT},
                                          messages=[{"role": "user", "content": item[f"{cond}_user"]}])
            if resp.stop_reason == "refusal":
                return {"verdict": None, "error": "refusal"}
            return parse("".join(b.text for b in resp.content if b.type == "text")) | {"model": resp.model}
        retry = (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError)
        return H.with_retries(once, retry_on=lambda e: isinstance(e, retry))
    return call


def cli_caller(system: str):
    cli = os.environ.get("CLAUDE_CLI", "claude")
    empty = Path(tempfile.mkdtemp(prefix="opus_empty_"))
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_CODE_") and k not in ("CLAUDECODE", "ANTHROPIC_BASE_URL")}
    cmd = [cli, "-p", "--model", MODEL, "--effort", EFFORT, "--tools", "", "--settings", '{"disableAllHooks":true}',
           "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence",
           "--system-prompt", system, "--output-format", "json"]

    def call(item, cond):
        def once():
            r = subprocess.run(cmd, input=item[f"{cond}_user"], capture_output=True, text=True, cwd=empty, env=env,
                               timeout=600)
            d = json.loads(r.stdout)
            if d.get("is_error"):
                raise RuntimeError("cli_error")
            return parse(d.get("result")) | {"model": ",".join((d.get("modelUsage") or {}).keys())}
        return H.with_retries(once, attempts=4)
    return call


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--via", choices=("api", "cli"), default="api")
    ap.add_argument("--cache", type=Path, default=Path("cache/opus"))
    ap.add_argument("--splits", default="test", help="the paper scored test prompts only")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    system = opus_system_prompt()
    call = api_caller(system) if a.via == "api" else cli_caller(system)
    items = H.load_inputs(a.inputs, splits=set(a.splits.split(",")), limit=a.limit)
    res = H.run_all([(i, c) for c in ("zs", "ex") for i in items], call, H.Cache(a.cache),
                    {"model": MODEL, "effort": EFFORT, "system": system}, a.workers)
    rows = []
    for c in ("zs", "ex"):
        for i in items:
            r = res[(i["split"], i["prompt_id"], c)]
            rows.append({"prompt_id": i["prompt_id"], "org": i["org"], "label": i["label"], "condition": c,
                         "verdict": r.get("verdict"), "confidence": r.get("confidence"), "p_malicious": r.get("p_malicious")})
    H.write_rows(a.out, rows)
    print(f"wrote {len(rows)} rows to {a.out}; unresolved:", sum(r["verdict"] is None for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
