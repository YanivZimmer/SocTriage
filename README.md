# RiskFit: local triage models versus hosted decision models

Code and released outputs for the paper

> **Can Small Defenders Triage Without Frontier Models? Open Local Models versus Hosted Decision Models for
> Human-on-the-Loop SOC Triage** (AICS 2027, AAAI-27 Workshop on AI for Cyber Security).
> Authors: Yaniv Zimmer, Yuval Goldberg, Noam Cohen. Contact: yaniv.zimmer@torq.io

The paper compares self-hosted trained models (RiskFit, cross-entropy, focal-Tversky, MaxSim, frozen-encoder
kNN, LightGBM, SetFit-LoRA, Qwen3-0.6B-LoRA) with hosted decision models (OpenAI Decisions API, TypeSafe Jev)
and frontier LLMs (Gemini 2.5 Pro, Claude Opus 5.5) on SOC alert triage. It scores each one by unreviewed
misses against false escalations, at default verdicts, at validation-tuned thresholds and under an 80%
confidence gate. Two datasets are used:

* **Private production alerts** (1,492 analyst-reviewed alerts from three tenants). The data owners did not
  permit release, so this repository holds no production data, alert text or predictions. The code that
  trains and evaluates the models on that data is included and runs on any data in the same schema
  (see [Production data](#production-data-not-reproducible)).
* **The public GUIDE benchmark** (12 organizations). Every GUIDE number in the paper can be reproduced from
  this repository: the dataset download, the split builder with fingerprint checks, the training and
  analysis code, and the hosted models' released per-prompt outputs.

## Repository layout

```
configs/
  guide_selection.yaml        pre-registered GUIDE organization selection and split rule, expected checksums
  guide_orgs.csv              the 12 organizations: counts and split fingerprints
  models/                     GUIDE recipes: riskfit_guide.yaml, ce_guide.yaml, lightgbm.yaml
  models/production/          production recipes: riskfit, riskfit_rho08, riskfit_lambda4, riskfit_beta025
                              (beta option), ce, focal_tversky, maxsim, knn, setfit_lora, qwen3_lora (.yaml)
riskfit/
  __init__.py                 package version
  serialization.py            normalized alert schema and the 21-field text renderer
  guide.py                    GUIDE incident aggregation, distinct-prompt collapse, selection, splits, fingerprints
  prompts.py                  hosted-model instructions, user messages, label-blind exemplar selector
  hosted.py                   shared client plumbing (cache, retries, output CSV)
  evaluation.py               operating points, gates, cost, crossovers, bootstrap, paired tests, Holm, reweighting
  evasion.py                  evasion perturbations P1-P4 and the R0 leakage control
  models/__init__.py          build_model(recipe): model factory keyed by model_type
  models/fastfit_lora.py      SecureBERT 2.0 + LoRA classifier (RiskFit / CE / focal-Tversky / MaxSim)
  models/fastfit/             vendored FastFit (__init__.py, modeling.py, train.py) with the RiskFit objective
                              and the balanced batch sampler
  models/tabular.py           LightGBM and its structured features
  models/baselines.py         frozen-SecureBERT kNN, SetFit-LoRA, Qwen3-0.6B-LoRA
scripts/
  download_guide.sh           download and checksum the public GUIDE dataset (not redistributed here)
  build_guide_splits.py       rebuild the 12-organization benchmark and verify fingerprints
  train_guide.py              train a local model per organization, write per-prompt predictions
  build_hosted_inputs.py      zero-shot and one-exemplar messages for the hosted models (no API call)
  run_decisions.py            OpenAI Decisions API client
  run_jev.py                  TypeSafe Jev client
  run_opus.py                 Claude Opus 5.5 client (Messages API, or the Claude Code CLI path used in the paper)
  analyze_guide.py            every GUIDE statistic in the paper
data/guide_outputs/           released hosted-model outputs on GUIDE (per prompt)
tests/smoke_test.py           end-to-end smoke test on one organization
SCRUB_REPORT.md               what was removed for release, and the scan results
LICENSE                       Apache License 2.0
requirements.txt              pinned dependencies
.gitignore                    keeps downloaded data, splits, predictions, caches and key files out of git
```

## Environment

Python 3.12 with the pinned versions in `requirements.txt`. These are the package versions of the
environments that produced the runs.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

On macOS, LightGBM also needs the OpenMP runtime (`brew install libomp`). The encoder models download
`cisco-ai/SecureBERT2.0-biencoder` (about 600 MB) from the Hugging Face Hub on first use. The production-only
baselines download `sentence-transformers/paraphrase-mpnet-base-v2` (SetFit) and `Qwen/Qwen3-0.6B-Base`.

Hardware and runtime, as measured for the paper:

| Step | Hardware | Time |
|---|---|---|
| Build the 12 GUIDE splits | laptop CPU | about 1 minute (reads 3.5 GB of CSV) |
| Train RiskFit or CE, 12 organizations | one NVIDIA A100 40 GB | about 3 GPU-hours per method (20-epoch ceiling, early stopping) |
| Train LightGBM, 12 organizations | 8 CPU cores | about 15 minutes |
| Analysis (2,000 bootstrap replicates) | laptop CPU | about 1 minute |
| Smoke test (one organization, tiny training subset) | laptop CPU or Apple GPU | a few minutes |
| Re-query hosted models (optional) | network | Decisions and Jev under USD 1 in total; Opus about 3.8M input tokens |

## Reproducing the GUIDE results

```bash
# 1. Data: public GUIDE (Kaggle "Microsoft Security Incident Prediction", dataset version 2), checksum-verified
scripts/download_guide.sh data/guide_raw

# 2. The 12-organization distinct-prompt benchmark. Every split fingerprint, the incident->prompt->split
#    membership table and the tie table must match the paper ("ALL CHECKS PASSED").
python scripts/build_guide_splits.py --source-dir data/guide_raw --out data/guide_splits --archive data/guide_raw/archive.zip
#    optional: re-run the whole pre-registered selection walk (93 organizations visited in hash order)
python scripts/build_guide_splits.py --source-dir data/guide_raw --out data/guide_splits_select --select

# 3. Local models (one training seed, 42), per organization
python scripts/train_guide.py --splits data/guide_splits/splits --method riskfit  --out data/predictions
python scripts/train_guide.py --splits data/guide_splits/splits --method ce       --out data/predictions
python scripts/train_guide.py --splits data/guide_splits/splits --method lightgbm --out data/predictions

# 4. Hosted-model inputs: byte-identical to the paper's requests (checks instruction hashes, exemplar ids)
python scripts/build_hosted_inputs.py --splits data/guide_splits/splits --out data/hosted_inputs.jsonl

# 5. All GUIDE statistics, using the released hosted outputs in data/guide_outputs/
python scripts/analyze_guide.py --predictions data/predictions --splits data/guide_splits/splits --out results/guide
# or, without retraining, with the paper's released local-model predictions:
python scripts/analyze_guide.py --predictions data/guide_predictions --splits data/guide_splits/splits --out results/guide
cat results/guide/paper_numbers.txt
```

Step 5 needs no API access. With the local predictions from the paper's runs, it reproduces all of the
paper's GUIDE numbers exactly. That includes the bootstrap intervals, because the bootstrap seed and the
resampling order are fixed. Retraining the local models (step 3) on different hardware or library builds can
move individual predictions slightly, so expect small differences in RiskFit, CE and LightGBM counts.

**Re-querying the hosted models (optional).** The released outputs are what the paper analyzes. To query the
models again, set the key in the environment (keys are never read from files) and write to a new directory:

```bash
OPENAI_API_KEY=...    python scripts/run_decisions.py --inputs data/hosted_inputs.jsonl --out out/decisions_guide_per_prompt.csv
TYPESAFE_API_KEY=...  python scripts/run_jev.py       --inputs data/hosted_inputs.jsonl --out out/jev_guide_per_prompt.csv
ANTHROPIC_API_KEY=... python scripts/run_opus.py      --inputs data/hosted_inputs.jsonl --out out/opus_guide_per_prompt.csv
python scripts/run_opus.py --via cli ...   # the Claude Code CLI path used in the paper (no temperature control)
python scripts/analyze_guide.py --hosted out --predictions data/predictions --splits data/guide_splits/splits --out results/guide_requery
```

Decisions (`gpt-6-luna`, queried 2026-10-29 and 2026-11-01) and Jev (`jev-1.14.0`, queried 2026-11-01) have
no dated snapshots, and Claude Opus 5.5 (2026-11-03, medium effort) was run once and is not deterministic,
so a re-query can differ from the released outputs. The Opus client does not enable server-side refusal
fallbacks, so every answer comes from `claude-opus-5-5`; a refusal is recorded as unresolved.

**Smoke test** (one organization, about 220 training prompts, a 16-prompt training subset):

```bash
python tests/smoke_test.py --source-dir data/guide_raw --org 77
```

### Released hosted-model outputs (`data/guide_outputs/`)

Public GUIDE identifiers and model outputs only. Prompt ids are `guide-<OrgId>-p<sha256(prompt text)[:16]>`;
labels are prompt majority labels (`false_positive`, `benign`, `malicious`).

| File | Rows | Columns |
|---|---|---|
| `decisions_guide_per_prompt.csv` | 2,158 (1,079 test + 1,079 validation) | exemplar id/status/score; for `ex` (one exemplar) and `zs` (zero-shot): verdict, confidence, q1 choice probabilities (`p_fp`, `p_benign`, `p_mal_q1`), q2 predicate probability (`p_mal_q2`) |
| `jev_guide_per_prompt.csv` | 2,158 | as above; `p_mal_q2` is the yes/no probability, plus input token counts |
| `opus_guide_per_prompt.csv` | 2,158 (1,079 test prompts x 2 conditions) | verdict, verbalized confidence, `p_malicious` |
| `decisions_jev_5ex_guide_per_prompt.csv` | 2,158 | follow-up, not in the current paper draft: Decisions and Jev with the five most similar training prompts as exemplars (`dec5_*`, `jev5_*`), the five exemplar ids, scores and labels, and the sha256 of each five-exemplar message |


The five-exemplar inputs are rebuilt, and checked byte-for-byte against the released message hashes, with
`python scripts/build_hosted_inputs.py --splits data/guide_splits/splits --k 5 --out data/hosted_inputs_k5.jsonl`.
The same clients send them. `analyze_guide.py` covers only the paper's conditions, so the five-exemplar
analysis is not wired in.

`data/guide_outputs/cybersecqwen4b_zeroshot_guide_per_prompt.csv` holds the per-prompt outputs of the open model `athena129/CyberSecQwen-4B` (Hugging Face revision `6c82bf13`, Apache-2.0), run zero-shot and locally on the 1,079 GUIDE test and 1,079 validation prompts. Each verdict's probability is the softmax of the summed log-probabilities of the three verdict labels, scored after the model's chat template with the decision models' instructions (no sampling). The scoring script is not included in this snapshot; the outputs can be fed to the analysis as an extra hosted-style method.

## Paper-to-code map

| Paper | What | Code |
|---|---|---|
| Methods: Tenant model | 21-field serialization, 1,024-token truncation | `riskfit/serialization.py`; `max_text_length` in the recipes |
| Methods: RiskFit objective, Eq. (1) | CE + gamma SupCon + lambda max(0, rho - R_soft), optional beta FPR_soft; balanced micro-batches of 4 with 9 accumulation steps | `riskfit/models/fastfit/modeling.py` (`lagrangian_recall_loss`, `SupConLoss`), `riskfit/models/fastfit/train.py` (`MaliciousBalancedBatchSampler`), `riskfit/models/fastfit_lora.py`, `configs/models/*.yaml` |
| Methods: Baselines | CE, focal-Tversky, MaxSim, kNN, LightGBM, SetFit-LoRA, Qwen3-0.6B-LoRA | `configs/models/production/*.yaml`, `riskfit/models/` |
| Methods: hosted models | Decisions, Jev, Opus questions and prompts; label-blind exemplar | `riskfit/prompts.py`, `scripts/run_*.py` |
| Methods: Operating points and gate | default verdict, validation-tuned threshold, test-quantile and deployable gates | `riskfit/evaluation.py` (`np_threshold`, `gate`, `tau_validation`) |
| Experimental setup: Public replication (GUIDE) | pre-registered selection, distinct-prompt unit, 70/15/15 split | `configs/guide_selection.yaml`, `riskfit/guide.py`, `scripts/build_guide_splits.py` |
| Metrics and statistics | bootstrap (2,000 replicates, seed 42), Wilcoxon / sign tests with Holm, exact McNemar | `riskfit/evaluation.py` (`bootstrap_gate_counts`, `paired`, `holm`, `mcnemar_exact`) |
| RQ1, GUIDE: default operating points | caught / false malicious; RiskFit vs LightGBM recall (Holm p = 0.024) | `analyze_guide.py` -> `full_coverage.csv`, `paired_tests.csv` |
| RQ1, GUIDE: thresholds and ranking | matched-FPR recall, ROC-AUC | `matched_fpr.csv`, `auc.csv`, `paired_tests.csv` |
| RQ2, GUIDE: gating on public data | gate misses / FE, cost at c = 10, P(local cheapest), deployable gate, Opus ties | `gate80.csv`, `frontier_cost.csv`, `bootstrap_*.csv`, `gate80_ties.csv` |
| Table: crossover, GUIDE columns | c* against each hosted model, reweighted to the production class ratio | `bootstrap_cost_crossovers.csv`, `crossovers_reweighted.csv` |
| RQ2, GUIDE: exemplar use | exemplar label correct 86%; followed 55% / 51% / 79% | `exemplar_following_test.csv` |
| Limitations: hosted-model agreement | score correlation of the two decision models | `hosted_agreement.csv` |
| RQ2: Evasive alerts (Table: evasion) | P1 benign-name masquerade, P2 command obfuscation, P3 fresh infrastructure, P4 all, R0 drop vendor response | `riskfit/evasion.py` (applied to private alerts; no data released) |
| Table: threshold, Bayes column | p(malicious) >= 1/(1+c) | `riskfit/evaluation.py` (`bayes_threshold`) |
| Production tables (Tables data, main, threshold, gate, evasion, weak; production crossovers) | private data | not reproducible; recipes and evaluation code included |

`results/guide/paper_numbers.txt` lists the GUIDE numbers quoted in the paper next to the section that uses them.

## Production data (not reproducible)

The production results (1,492 alerts, three tenants; Tables "data", "main", "threshold", "gate", "evasion",
"weak", the production columns of the crossover table, the forward-in-time test and the weak-supervision
study) cannot be reproduced from this repository. The alerts, labels and per-alert predictions are not
released, and neither are trained adapters. What is released:

* **Schema.** Training records are JSON objects in the normalized schema of `riskfit/serialization.py`
  (`NormalizedAlert`), plus a training envelope: `alert_id`, `label` (one of `False Positive`,
  `True Positive - Benign`, `True Positive - Malicious`) and `label_source` (`gold` = the analyst corrected
  the production LLM's verdict, `silver` = the analyst confirmed it). The renderer emits these 21 fields, in
  this order: alert name, severity, source, category, risk score, observables (with threat-intelligence
  enrichment: AV ratio, threat label, verdict, reputation), objective, description, hostname, platform, user,
  file name, file path, process chain (process <- parent <- grandparent), command line, detection engines,
  vendor response, trigger IOC, files accessed, MITRE tactics and MITRE techniques. The label, analyst
  comment, status and LLM verdict are never rendered.
* **Recipes.** `configs/models/production/` holds every production recipe, including the 2:1 weighting of
  analyst-corrected labels (`apply_weighting`). Put a fold in `splits/<name>/{train,val,test}.jsonl` with a
  `benchmark_split_manifest.json` (see `riskfit/guide.py::write_org`). Then
  `scripts/train_guide.py --config configs/models/production/<recipe>.yaml` trains and predicts it.
  The bundled split loader checks the manifest hashes.
* **Evaluation.** `riskfit/evaluation.py` implements every production statistic: gates, cost, crossovers,
  bootstrap, McNemar with Holm, and the Bayes threshold. `riskfit/evasion.py` holds the perturbations.

Not included, because the code is tied to private infrastructure or data: the production data export and
labelling pipeline, the Gemini 2.5 Pro client used on production alerts (the zero-shot and exemplar prompts
are the same as in `riskfit/prompts.py`), the forward-in-time and weak-supervision drivers, the near-duplicate
and leakage audits, and the cloud job launchers.

## Notes

* GUIDE is distributed by Microsoft on Kaggle under its own terms; it is not redistributed here.
* `riskfit/models/fastfit/` is a modified copy of IBM FastFit (Apache License 2.0).
* The string `X4-GUIDE-org-` in the organization order is intentional: it is the pre-registered hash salt,
  and changing it changes the selection.

## Released local-model predictions

`data/guide_predictions/{riskfit,ce,lightgbm}/org_<id>.jsonl` hold the paper's per-prompt GUIDE test and validation predictions (6,474 rows; fields: `alert_id`, `split`, `expected_label`, `predicted_label`, `confidence`, `class_probs`). With them, step 5 reproduces the paper's GUIDE tables without retraining.

## License

Apache License 2.0 (see `LICENSE`).
