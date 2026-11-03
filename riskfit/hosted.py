"""Shared plumbing for the hosted-model clients (GUIDE only): inputs, response cache, retries, output CSV.

API keys are read only from environment variables by each client; nothing here reads or writes keys.
Responses are cached by sha256(model, questions/system, user text, split, prompt id, condition), so a rerun
never repeats a paid call.
"""

from __future__ import annotations

import concurrent.futures as cf
import csv
import hashlib
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Callable


def require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"set the {name} environment variable (keys are never read from files)")
    return v


def load_inputs(path: Path, splits: set[str] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    items = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if splits:
        items = [i for i in items if i["split"] in splits]
    return items[:limit] if limit else items


def request_key(config: Any, item: dict[str, Any], cond: str) -> str:
    payload = {"config": config, "user": item[f"{cond}_user"], "split": item["split"], "prompt_id": item["prompt_id"],
               "cond": cond}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class Cache:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self.path(key)
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, key: str, rec: dict) -> None:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(rec))
        tmp.replace(p)


def with_retries(fn: Callable[[], dict], attempts: int = 8, retry_on: Callable[[Exception], bool] = lambda e: True) -> dict:
    """Exponential backoff (max 8 attempts) on transient failures; other errors are recorded, not raised."""
    for k in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - record the error category only
            if k == attempts - 1 or not retry_on(e):
                return {"verdict": None, "error": type(e).__name__}
            time.sleep(min(60, 2 ** (k + 1)) + random.random())
    return {"verdict": None, "error": "exhausted"}


def run_all(jobs: list[tuple[dict, str]], call: Callable[[dict, str], dict], cache: Cache, config: Any,
            workers: int) -> dict[tuple[str, str, str], dict]:
    """Run (item, condition) jobs through ``call`` with caching; returns {(split, prompt_id, cond): result}."""
    out = {}

    def one(job):
        item, cond = job
        key = request_key(config, item, cond)
        hit = cache.get(key)
        if hit is None:
            hit = call(item, cond)
            if hit.get("verdict") is not None:
                cache.put(key, hit)
        return (item["split"], item["prompt_id"], cond), hit

    with cf.ThreadPoolExecutor(workers) as ex:
        for k, v in ex.map(one, jobs):
            out[k] = v
    return out


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows({k: ("" if v is None else v) for k, v in r.items()} for r in rows)
