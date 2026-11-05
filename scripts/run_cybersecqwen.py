#!/usr/bin/env python3
"""CyberSecQwen-4B, zero-shot and run locally, on the GUIDE prompts: the scoring behind
data/guide_outputs/cybersecqwen4b_zeroshot_guide_per_prompt.csv (paper: "scoring details are in the code release").

Model: athena129/CyberSecQwen-4B (Hugging Face), revision 6c82bf137b2c7446b4c17ba83f8090544702b077, Apache-2.0,
a fine-tune of Qwen3-4B-Instruct-2507. Safetensors only, loaded with trust_remote_code=False.

Prompt. One user turn, rendered with the model's own chat template (add_generation_prompt=True):
riskfit.prompts.open_llm_user_prompt(zs_user), i.e. Opus's instructions (both decision questions) with the JSON
output block replaced by "answer with a single verdict label", then the byte-identical zero-shot user message the
hosted models received, then "Which verdict applies to the TARGET ALERT? Answer with exactly one word:
false_positive, benign, or malicious." No exemplar, no system message, no sampling.

Scoring. For each label in (false_positive, benign, malicious) the label's full token sequence (canonical
tokenization, no leading space) is teacher-forced after the prompt and its token log-probabilities are summed.
p_fp, p_benign, p_malicious = softmax of the three sums (float64); verdict = argmax; confidence = the largest
probability (the gate's confidence); p_malicious ranks alerts. first_token_label_mass is the next-token
probability mass on the three labels' first tokens (a diagnostic: how much of the model's mass the three labels
take). Greedy decoding agreed with the scored argmax on 20 of 20 sampled prompts in the paper's run.

Paper run: transformers 4.54.1, torch 2.4.1, bf16 on Apple MPS, about 1.3 s per distinct prompt (41 minutes for
1,843 distinct texts). transformers 4.54 does not read the v5-style "rope_parameters" block of this model's
config and would silently use rope_theta = 1e4, so rope_theta is set explicitly to the config's 5,000,000; the
config's list-valued "extra_special_tokens" is overridden with {} (those tokens are already special tokens in
tokenizer.json). Weights and tokenizer files are unmodified.

Usage:
  python scripts/build_hosted_inputs.py --splits data/guide_splits/splits --out data/hosted_inputs.jsonl
  huggingface-cli download athena129/CyberSecQwen-4B --revision 6c82bf137b2c7446b4c17ba83f8090544702b077 \\
      --local-dir models/CyberSecQwen-4B
  # check the prompts against the released token counts (needs only the tokenizer files, no weights):
  python scripts/run_cybersecqwen.py --inputs data/hosted_inputs.jsonl --model-dir models/CyberSecQwen-4B --check-tokens
  # score (about 8-11 GB of accelerator memory in bf16):
  python scripts/run_cybersecqwen.py --inputs data/hosted_inputs.jsonl --model-dir models/CyberSecQwen-4B \\
      --device mps --out out/cybersecqwen4b_zeroshot_guide_per_prompt.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from riskfit.prompts import VERDICTS, open_llm_user_prompt  # noqa: E402

REVISION = "6c82bf137b2c7446b4c17ba83f8090544702b077"
COLUMNS = ["split", "org", "id", "y", "verdict", "confidence", "p_fp", "p_benign", "p_malicious", "logp_fp",
           "logp_benign", "logp_malicious", "first_token_label_mass", "n_prompt_tokens"]
RELEASED = ROOT / "data" / "guide_outputs" / "cybersecqwen4b_zeroshot_guide_per_prompt.csv"


def load_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False, local_files_only=True, extra_special_tokens={})
    for t, i in (("<|im_start|>", 151644), ("<|im_end|>", 151645), ("<|endoftext|>", 151643)):
        assert tok("x" + t + "y", add_special_tokens=False).input_ids.count(i) == 1, t
    return tok


def load_model(model_dir: Path, device: str):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    raw = json.loads((model_dir / "config.json").read_text())
    assert "auto_map" not in raw, "config requires custom code"
    assert raw["architectures"] == ["Qwen3ForCausalLM"] and raw["model_type"] == "qwen3"
    bad = [p.name for p in model_dir.iterdir() if p.suffix in (".py", ".bin", ".pt", ".pth", ".pkl", ".ckpt")]
    assert not bad, f"unexpected files {bad}"
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=False, local_files_only=True)
    rp = raw.get("rope_parameters")
    if rp is not None:  # see the module docstring
        assert rp == {"rope_theta": 5000000, "rope_type": "default"}, rp
        cfg.rope_theta = float(rp["rope_theta"])
    model = AutoModelForCausalLM.from_pretrained(model_dir, config=cfg, torch_dtype=torch.bfloat16, use_safetensors=True,
                                                 trust_remote_code=False, local_files_only=True, low_cpu_mem_usage=True)
    return model.to(device).eval()


def render(tok, model_dir: Path, text: str) -> str:
    """The model's chat template (chat_template.jinja), one user turn, generation prompt appended."""
    import jinja2

    p = model_dir / "chat_template.jinja"
    if p.exists():
        s = jinja2.Environment(keep_trailing_newline=True).from_string(p.read_text()).render(
            messages=[{"role": "user", "content": text}], add_generation_prompt=True)
    else:
        s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return s


def score(model, tok, prompt: str, label_ids: list[list[int]], device: str):
    import torch

    with torch.inference_mode():
        ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        n = ids.shape[1]
        out = model(ids, use_cache=True)
        first = torch.log_softmax(out.logits[0, -1].float(), -1)
        pkv = out.past_key_values
        lps, first_lp = [], []
        for lab in label_ids:
            lp = float(first[lab[0]])
            first_lp.append(lp)
            if len(lab) > 1:
                o2 = model(torch.tensor([lab[:-1]], device=device), past_key_values=pkv, use_cache=True)
                ls = torch.log_softmax(o2.logits[0].float(), -1)
                lp += sum(float(ls[j, lab[j + 1]]) for j in range(len(lab) - 1))
                pkv.crop(n)
            lps.append(lp)
    return n, lps, first_lp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True, help="data/hosted_inputs.jsonl (scripts/build_hosted_inputs.py)")
    ap.add_argument("--model-dir", type=Path, required=True, help=f"local snapshot of athena129/CyberSecQwen-4B@{REVISION[:8]}")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device", default="cuda", help="cuda, mps or cpu")
    ap.add_argument("--splits", default="test,val")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--check-tokens", action="store_true",
                    help="only render and tokenize the prompts and compare n_prompt_tokens with the released outputs")
    a = ap.parse_args()

    which = set(a.splits.split(","))
    items = [json.loads(line) for line in a.inputs.open() if line.strip()]
    items = [i for i in items if i["split"] in which][: a.limit]
    tok = load_tokenizer(a.model_dir)
    label_ids = [tok(v, add_special_tokens=False).input_ids for v in VERDICTS]
    assert len({i[0] for i in label_ids}) == 3, "labels share a first token"

    if a.check_tokens:
        ref = {(r["split"], r["id"]): int(r["n_prompt_tokens"]) for r in csv.DictReader(RELEASED.open(newline=""))}
        bad = 0
        for i in items:
            n = len(tok(render(tok, a.model_dir, open_llm_user_prompt(i["zs_user"])), add_special_tokens=False).input_ids)
            bad += n != ref[(i["split"], i["prompt_id"])]
        print(f"[{'PASS' if not bad else 'FAIL'}] prompt token counts equal the released outputs ({len(items)} prompts, {bad} differ)")
        return 1 if bad else 0

    import torch

    assert a.out, "--out is required unless --check-tokens"
    model = load_model(a.model_dir, a.device)
    cache: dict[str, tuple] = {}  # identical prompt texts are scored once
    rows = []
    for k, i in enumerate(items):
        prompt = render(tok, a.model_dir, open_llm_user_prompt(i["zs_user"]))
        if prompt not in cache:
            cache[prompt] = score(model, tok, prompt, label_ids, a.device)
            if len(cache) % 25 == 0:
                gc.collect()
                if a.device == "mps":
                    torch.mps.empty_cache()
        n, lps, first_lp = cache[prompt]
        p = torch.softmax(torch.tensor(lps, dtype=torch.float64), 0).tolist()
        v = max(range(3), key=lambda j: p[j])
        rows.append({"split": i["split"], "org": i["org"], "id": i["prompt_id"], "y": i["label"], "verdict": VERDICTS[v],
                     "confidence": p[v], "p_fp": p[0], "p_benign": p[1], "p_malicious": p[2], "logp_fp": lps[0],
                     "logp_benign": lps[1], "logp_malicious": lps[2],
                     "first_token_label_mass": sum(float(torch.tensor(x).exp()) for x in first_lp), "n_prompt_tokens": n})
        if k % 100 == 0:
            print(f"{k + 1}/{len(items)} scored ({len(cache)} distinct prompts)", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
