#!/usr/bin/env python3
"""End-to-end smoke test on one GUIDE organization with a tiny training subset (no API calls).

Steps: (1) rebuild the organization's split and check its fingerprint against configs/guide_orgs.csv;
(2) build the hosted-model inputs and check prompt ids, exemplars and instruction hashes against the released
outputs; (3) train RiskFit and CE for one epoch on 16 training prompts and LightGBM on the full training split
(skipped with a stand-in when LightGBM cannot load, e.g. macOS without libomp),
predicting the organization's full validation and test splits; (4) run the GUIDE analysis for that
organization with the released hosted outputs and the released MaxSim, kNN and Tiny Decision predictions
(200 bootstrap replicates).

Needs the GUIDE CSVs (scripts/download_guide.sh) and the SecureBERT 2.0 weights (downloaded from the
Hugging Face Hub on first use, about 600 MB; set HF_HUB_OFFLINE=1 if they are already cached).
Runtime: about 1 minute to read GUIDE plus a few minutes of CPU/MPS training.

Usage: python tests/smoke_test.py --source-dir data/guide_raw [--org 77] [--workdir /tmp/riskfit_smoke]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, required=True)
    ap.add_argument("--org", default="77")
    ap.add_argument("--workdir", type=Path, default=Path(tempfile.mkdtemp(prefix="riskfit_smoke_")))
    ap.add_argument("--skip-encoders", action="store_true", help="skip RiskFit/CE (no transformer download)")
    a = ap.parse_args()
    w = a.workdir
    run("scripts/build_guide_splits.py", "--source-dir", str(a.source_dir), "--out", str(w / "splits_out"), "--orgs", a.org)
    splits = str(w / "splits_out" / "splits")
    run("scripts/build_hosted_inputs.py", "--splits", splits, "--orgs", a.org, "--out", str(w / "hosted_inputs.jsonl"))
    preds = str(w / "predictions")
    for m in ("riskfit", "ce"):
        if a.skip_encoders:
            continue
        run("scripts/train_guide.py", "--splits", splits, "--orgs", a.org, "--method", m, "--out", preds,
            "--epochs", "1", "--max-train", "16")
    try:
        import lightgbm  # noqa: F401  (on macOS needs the OpenMP runtime: brew install libomp)
        have_lgbm = True
    except Exception as e:  # noqa: BLE001
        have_lgbm = False
        print(f"LightGBM unavailable ({type(e).__name__}); its predictions are replaced by a stand-in copy", flush=True)
    if have_lgbm:
        run("scripts/train_guide.py", "--splits", splits, "--orgs", a.org, "--method", "lightgbm", "--out", preds)
    # stand-ins keep the analysis runnable when a model family was skipped (smoke runs only)
    src = next((m for m in ("riskfit", "ce", "lightgbm") if (w / "predictions" / m / f"org_{a.org}.jsonl").exists()), None)
    if src is None:
        raise SystemExit("no model could be trained")
    for m in ("riskfit", "ce", "lightgbm"):
        f = w / "predictions" / m / f"org_{a.org}.jsonl"
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes((w / "predictions" / src / f"org_{a.org}.jsonl").read_bytes())
            print(f"stand-in: {m} <- {src}", flush=True)
    for m in ("maxsim", "knn", "tiny_decision"):  # not retrained here: the paper's released predictions
        f = w / "predictions" / m / f"org_{a.org}.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes((ROOT / "data" / "guide_predictions" / m / f"org_{a.org}.jsonl").read_bytes())
    run("scripts/analyze_guide.py", "--predictions", preds, "--splits", splits, "--orgs", a.org,
        "--out", str(w / "analysis"), "--n-boot", "200")
    print(f"SMOKE TEST PASSED (outputs in {w})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
