# RiskFit: local triage models versus hosted decision models

Code, released per-prompt outputs and aggregate results for the paper

> **Can Small Defenders Triage Without Frontier Models? Open Local Models versus Hosted Decision Models for
> Human-on-the-Loop SOC Triage** (AICS 2027, AAAI-27 Workshop on AI for Cyber Security).
> Authors: Yaniv Zimmer, Yuval Goldberg, Noam Cohen. Contact: yaniv.zimmer@torq.io

The paper scores SOC alert triage by unreviewed misses against false escalations, with one miss costing as much
as `c` false escalations, at default verdicts, at validation-tuned thresholds and under an 80% confidence gate.
It compares:

* **Local models trained per tenant**: RiskFit (SecureBERT 2.0 + LoRA with a hinge on soft malicious recall),
  cross-entropy (CE), focal-Tversky, MaxSim, frozen-SecureBERT kNN, LightGBM, SetFit-LoRA and Qwen3-0.6B-LoRA.
* **Hosted decision models**: the OpenAI Decisions API (`gpt-6-luna`) and, on GUIDE, TypeSafe Jev (`jev-1.14.0`),
  zero-shot and with one retrieved exemplar; on GUIDE also with five exemplars.
* **Generative LLMs**: Gemini 2.5 Pro on production alerts, Claude Opus 5.5 on GUIDE (zero-shot and with one
  exemplar), and CyberSecQwen-4B, an open security LLM run locally and zero-shot on GUIDE.

Two datasets are used:

* **Private production alerts** (1,492 analyst-reviewed alerts from three tenants, A/B/C). The data owners
  permitted publication of aggregate results only. This repository releases those **aggregate results** (per
  tenant or pooled; `aggregates/production/`) and the code that trains and evaluates the models on data in the
  same schema, but no alert text, labels, per-alert predictions, customer identifiers or trained adapters.
* **The public GUIDE benchmark** (12 organizations). Every GUIDE number in the paper is reproducible from this
  repository: the dataset download, the split builder with fingerprint checks, the training and analysis code,
  the hosted models' per-prompt outputs and the paper's local-model per-prompt predictions.

## What is and is not released

| Released | Not released |
|---|---|
| All code for the GUIDE comparisons (splits, local models, hosted-model clients, CyberSecQwen-4B scoring, analysis, figures) | Production alerts, alert text, raw alerts |
| Per-prompt GUIDE outputs of every hosted / open-LLM condition: Decisions API with 0 / 1 / 5 exemplars, Jev with 0 / 1 / 5 exemplars, Opus with 0 / 1, CyberSecQwen-4B zero-shot | Per-alert production labels, predictions or confidences |
| The paper's per-prompt GUIDE predictions of RiskFit, CE and LightGBM | Trained adapters and checkpoints |
| Aggregate production metrics behind every production table, figure and text number (per tenant or pooled) | Customer names, hosts, users, IPs, internal identifiers |
| Training recipes for every production model, the evaluation code and the evasion perturbations | The production export / labelling pipeline, the Gemini client, the cloud launchers |
| A snapshot of the GUIDE analysis output (`aggregates/guide/`) | The GUIDE dataset itself (download script provided) |

**Results not in this snapshot** (experiments still running; the paper draft marks them as pending):

* `TODO (pending results)` **Recall targeting** (RQ1, "Recall targeting": the rho sweep rho in {0.7, 0.8, 0.9, 0.95}
  with extra seeds, production and GUIDE). No code, outputs or aggregates for it are included yet.
* `TODO (pending results)` **Extra GUIDE baselines** beyond those listed above. No outputs are included yet.

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
  prompts.py                  hosted-model instructions, user messages, label-blind exemplar selector, open-LLM prompt
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
  build_hosted_inputs.py      zero-shot, one- and five-exemplar messages for the hosted models (no API call)
  run_decisions.py            OpenAI Decisions API client
  run_jev.py                  TypeSafe Jev client
  run_opus.py                 Claude Opus 5.5 client (Messages API, or the Claude Code CLI path used in the paper)
  run_cybersecqwen.py         CyberSecQwen-4B zero-shot label scoring (local; --check-tokens needs no weights)
  analyze_guide.py            every GUIDE statistic in the paper -> paper_numbers.txt
  make_figures.py             Figures 2 and 3 from released CSVs
data/guide_outputs/           released per-prompt outputs of the hosted models and CyberSecQwen-4B on GUIDE
data/guide_predictions/       the paper's per-prompt GUIDE predictions of RiskFit, CE and LightGBM
aggregates/production/        aggregate production results (tenant A/B/C or pooled), one CSV per table/analysis
aggregates/guide/             snapshot of analyze_guide.py's output on the released predictions (12 methods)
tests/smoke_test.py           end-to-end smoke test on one organization (trains on 16 prompts)
tests/test_release.py         release checks without training or downloads
SCRUB_REPORT.md               what was removed for release, and the scan results
LICENSE                       Apache License 2.0
requirements.txt              pinned dependencies
.gitignore                    keeps downloaded data, splits, predictions, caches, figures and key files out of git
```

## Environment

Python 3.12 with the pinned versions in `requirements.txt` (the versions of the environments that produced the runs).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

On macOS, LightGBM also needs the OpenMP runtime (`brew install libomp`). The encoder models download
`cisco-ai/SecureBERT2.0-biencoder` (about 600 MB) from the Hugging Face Hub on first use. The production-only
baselines download `sentence-transformers/paraphrase-mpnet-base-v2` (SetFit) and `Qwen/Qwen3-0.6B-Base`.
Re-scoring CyberSecQwen-4B needs `athena129/CyberSecQwen-4B` (about 8 GB, bf16).

| Step | Hardware | Time |
|---|---|---|
| Build the 12 GUIDE splits | laptop CPU | about 1 minute (reads 3.5 GB of CSV) |
| Train RiskFit or CE, 12 organizations | one NVIDIA A100 40 GB | about 3 GPU-hours per method (20-epoch ceiling, early stopping) |
| Train LightGBM, 12 organizations | 8 CPU cores | about 15 minutes |
| Analysis (12 methods, 2,000 bootstrap replicates) | laptop CPU | about 1 minute |
| Figures from released CSVs | laptop CPU | seconds |
| Score CyberSecQwen-4B (optional) | Apple M-series GPU (MPS) or one GPU | about 41 minutes (1,843 distinct prompts) |
| Re-query hosted models (optional) | network | Decisions and Jev about USD 1 in total including five exemplars; Opus about 3.8M input tokens |

## Reproducing the GUIDE results

```bash
# 1. Data: public GUIDE (Kaggle "Microsoft Security Incident Prediction", dataset version 2), checksum-verified
scripts/download_guide.sh data/guide_raw

# 2. The 12-organization distinct-prompt benchmark. Every split fingerprint, the incident->prompt->split
#    membership table and the tie table must match the paper ("ALL CHECKS PASSED").
python scripts/build_guide_splits.py --source-dir data/guide_raw --out data/guide_splits --archive data/guide_raw/archive.zip
#    optional: re-run the whole pre-registered selection walk (93 organizations visited in hash order)
python scripts/build_guide_splits.py --source-dir data/guide_raw --out data/guide_splits_select --select

# 3. Local models (one training seed, 42), per organization -- or skip and use data/guide_predictions/
python scripts/train_guide.py --splits data/guide_splits/splits --method riskfit  --out data/predictions
python scripts/train_guide.py --splits data/guide_splits/splits --method ce       --out data/predictions
python scripts/train_guide.py --splits data/guide_splits/splits --method lightgbm --out data/predictions

# 4. Hosted-model inputs: byte-identical to the paper's requests (instruction hashes, exemplar ids, and for k=5
#    the sha256 of every five-exemplar message are checked against the released outputs)
python scripts/build_hosted_inputs.py --splits data/guide_splits/splits --out data/hosted_inputs.jsonl
python scripts/build_hosted_inputs.py --splits data/guide_splits/splits --k 5 --out data/hosted_inputs_k5.jsonl

# 5. All GUIDE statistics (12 methods), using the released outputs in data/guide_outputs/
python scripts/analyze_guide.py --predictions data/guide_predictions --splits data/guide_splits/splits --out results/guide
cat results/guide/paper_numbers.txt          # every GUIDE number in the paper, grouped by table / figure / paragraph
diff -r results/guide aggregates/guide        # identical to the released snapshot

# 6. Figures 2 and 3 (GUIDE panels from step 5 or from the snapshot; production panels from aggregates/production)
python scripts/make_figures.py --guide results/guide --out figures
```

Step 5 needs no API access. With the paper's local-model predictions (`data/guide_predictions/`) it reproduces
every GUIDE number in the paper exactly, including bootstrap intervals (the bootstrap seed and resampling order are
fixed, and the resampling does not depend on which methods are analyzed). Retraining the local models (step 3) on
other hardware or library builds can move individual predictions slightly, so expect small differences in the
RiskFit, CE and LightGBM counts. `--five-exemplar none` / `--open-llm none` drop the follow-up conditions.

**Re-querying the hosted models (optional).** The released outputs are what the paper analyzes. Keys are read only
from the environment:

```bash
OPENAI_API_KEY=...    python scripts/run_decisions.py --inputs data/hosted_inputs.jsonl --out out/decisions_guide_per_prompt.csv
TYPESAFE_API_KEY=...  python scripts/run_jev.py       --inputs data/hosted_inputs.jsonl --out out/jev_guide_per_prompt.csv
ANTHROPIC_API_KEY=... python scripts/run_opus.py      --inputs data/hosted_inputs.jsonl --out out/opus_guide_per_prompt.csv
python scripts/run_opus.py --via cli ...   # the Claude Code CLI path used in the paper (no temperature control)
# five exemplars: the same clients on data/hosted_inputs_k5.jsonl (their ex_* columns are then the 5-exemplar condition)
python scripts/analyze_guide.py --hosted out --predictions data/guide_predictions --splits data/guide_splits/splits --out results/guide_requery
```

Decisions (`gpt-6-luna`, queried 2026-10-29 to 2026-11-03) and Jev (`jev-1.14.0`, 2026-11-01 and 2026-11-03) have
no dated snapshots, and Claude Opus 5.5 (2026-11-03, medium effort) was run once and is not deterministic, so a
re-query can differ from the released outputs. The Opus client does not enable server-side refusal fallbacks;
a refusal is recorded as unresolved.

**CyberSecQwen-4B scoring.** `scripts/run_cybersecqwen.py` documents and implements the scoring used for the
paper (the paper's "scoring details are in the code release"): `athena129/CyberSecQwen-4B` at revision
`6c82bf13`, one user turn in the model's chat template holding the decision models' instructions, the
byte-identical zero-shot alert message and a one-word question; each of the three verdict labels is
teacher-forced and its token log-probabilities summed; `p(malicious)` is the softmax over the three sums, the
verdict is the argmax and the gate confidence is the largest probability. No sampling. `--check-tokens`
re-renders all 2,158 prompts and compares their token counts with the released file using only the tokenizer.

### Released per-prompt outputs (`data/guide_outputs/`)

Public GUIDE identifiers and model outputs only. Prompt ids are `guide-<OrgId>-p<sha256(prompt text)[:16]>`;
labels are prompt majority labels (`false_positive`, `benign`, `malicious`).

| File | Rows | Paper conditions | Columns |
|---|---|---|---|
| `decisions_guide_per_prompt.csv` | 2,158 (1,079 test + 1,079 validation) | Decisions zero-shot, + 1 exemplar | exemplar id/status/score; for `ex` and `zs`: verdict, confidence, choice probabilities (`p_fp`, `p_benign`, `p_mal_q1`), predicate probability (`p_mal_q2`) |
| `jev_guide_per_prompt.csv` | 2,158 | Jev zero-shot, + 1 exemplar | as above; `p_mal_q2` is the yes/no probability, plus input token counts |
| `decisions_jev_5ex_guide_per_prompt.csv` | 2,158 | Decisions + 5 exemplars, Jev + 5 exemplars | the five exemplar ids, scores and labels, the sha256 of each five-exemplar message, `dec5_*` and `jev5_*` outputs |
| `opus_guide_per_prompt.csv` | 2,158 (1,079 test prompts x 2 conditions) | Opus zero-shot, + 1 exemplar | verdict, verbalized confidence, `p_malicious` |
| `cybersecqwen4b_zeroshot_guide_per_prompt.csv` | 2,158 | CyberSecQwen-4B zero-shot | verdict, confidence, the three label probabilities and summed log-probabilities, first-token label mass, prompt tokens |

`data/guide_predictions/{riskfit,ce,lightgbm}/org_<id>.jsonl` hold the paper's per-prompt GUIDE test and
validation predictions (6,474 rows; `alert_id`, `split`, `expected_label`, `predicted_label`, `confidence`,
`class_probs`).

## Paper-to-repository map

Numbering follows the paper draft (Table 1 = "the answer at a glance", Figure 1 = the setting diagram).
"GUIDE" rows are reproduced by `scripts/analyze_guide.py` (file names are its outputs, also in `aggregates/guide/`;
`paper_numbers.txt` lists each number under the table or paragraph that quotes it). "Production" rows point to
`aggregates/production/`.

| Paper element | What | Repository file(s) |
|---|---|---|
| Table 1, GUIDE column | gate and full-coverage cost at c=10, crossovers 125 / 55 | GUIDE: `frontier_cost.csv`, `full_coverage.csv`, `bootstrap_cost_crossovers.csv` |
| Table 1, production column | 118 / 98, 154, 234, 233, crossovers 14 / 16 | `gate80_test_quantile.csv`, `default_verdicts.csv` (`cost_c10`), `one_to_one_vs_decisions_exemplar.csv` |
| Table 2 | alerts per tenant and verdict, corrections (malicious) | `data_summary.csv` |
| Setup text | 964 uncorrected labels, 943 equal to the LLM's export-time verdict | `data_summary.csv` |
| Table 3 (GUIDE thresholds) | validation-tuned @RiskFit / 1% / 2% / 5%, all seven rows, bold = lowest cost | GUIDE: `matched_fpr.csv` (pooled rows) |
| Figure 2, left | GUIDE default verdicts, 12 methods | GUIDE: `full_coverage.csv`; `scripts/make_figures.py` |
| Figure 2, right | production default verdicts, 13 methods (CE-A100 50/28 in the caption) | `default_verdicts.csv`; `scripts/make_figures.py` |
| RQ1 GUIDE text | Holm p = 0.024 / 0.008 / 0.003, five-exemplar costs 1,381 / 1,173, ROC-AUC | GUIDE: `paired_tests.csv`, `matched_fpr.csv`, `auc.csv` |
| Table 4 (GUIDE gate) | misses / FE, c* vs RiskFit and CE, ROC-AUC mean +- std | GUIDE: `frontier_cost.csv`, `bootstrap_cost_crossovers.csv`, `auc.csv` |
| RQ2 GUIDE text | cheapest shares (77% / 23%; 100% at c=10, > 99.5% at c=20 with 5-ex. or CyberSecQwen), Opus dominance 84% / 60%, intervals 3-15 and 5-19, c* 125 [30, inf), deployable 184 | GUIDE: `bootstrap_min_cost_probability.csv`, `summary.json`, `bootstrap_gate80.csv`, `bootstrap_cost_crossovers.csv` |
| RQ2 "Exemplars" | 86% correct, following 55 / 51 / 79% (45 / 43 / 45% zero-shot), 85 / 68% with five, -74% / -36% FE, neighbor vote 410/75 and 386/68 | GUIDE: `exemplar_following_test.csv`, `frontier_cost.csv`, `neighbor_vote.csv` |
| Figure 3, left + caption | GUIDE cost curves; reweighted crossovers 896 / 695 | GUIDE: `frontier_cost.csv`, `crossovers_reweighted.csv`; `scripts/make_figures.py` |
| Figure 3, right + caption | production cost curves; crossovers 14 / 16 / 11 and 115 / 117 / 78 | `gate80_test_quantile.csv`, `one_to_one_vs_decisions_exemplar.csv`; `scripts/make_figures.py` |
| RQ1 production text | hinge vs CE-A100 (+6 / +36; 2.3 +- 3.5 and 15 +- 20 over seeds), kappa .586 / .598, ROC-AUC and PR-AUC per tenant, thresholds at RiskFit's FPR (kNN 55/48, RiskFit 56/93, CE 56/79, Decisions 54/73 and 27/79), kNN @5% cost 204 | `default_verdicts.csv`, `seed_variance.csv`, `ranking_auc.csv`, `thresholds.csv` |
| Table 5 (production gate) | misses / FE pooled, Pareto set, per-tenant misses | `gate80_test_quantile.csv` (`pareto_pooled`, per-tenant rows) |
| RQ2 production text | 99.97% local cheapest, MaxSim / rho=0.8 intervals, one-to-one shares 89 / 95.5 / 93 / 99 / 42 / 67%, crossover 21 [11, 71] | `gate80_min_cost_probability.csv`, `gate80_test_quantile.csv`, `one_to_one_vs_decisions_exemplar.csv`, `gate80_crossovers_bootstrap.csv` |
| RQ2 "Seeds, deployable gate" | 103 +- 23, 106 +- 30, 90 +- 46 vs 155; deployable 6/26, 98%, automation 68 / 75 / 82% | `seed_deployable_cost.csv`, `seed_variance.csv`, `deployable_gate.csv`, `one_to_one_vs_decisions_exemplar.csv` |
| RQ2 "Later alerts" (tenant A) | 433 alerts, full coverage and gates of RiskFit / CE / LightGBM / Decisions, c* 55, FE rate .045 -> .18, retrieval 66% vs 38%, carried-forward automation | `later_alerts_tenantA_full_coverage.csv`, `later_alerts_tenantA_gates.csv`, `later_alerts_tenantA_coverage_curve.csv`, `later_alerts_local_models.csv`, `exemplar_retrieval.csv`, `facts.csv` |
| Table 6 (evasion) | misses under P1-P4 and R0, paired intervals, inputs changed 29 / 24 / 30 / 47 | `evasion.csv`, `one_to_one_vs_decisions_exemplar.csv` (perturbed conditions) |
| Table 7 + RQ3 text | weak vs 25%-only vs full; low-budget training counts (0 / 2 / 14 of 27); matched-fold comparison; tenant B over-calling (~330 of 471 vs 26) | `weak_supervision.csv`, `weak_vs_revealed_kappa_matched_folds.csv`, `weak_vs_revealed_cost_matched_folds_k25.csv`, `data_summary.csv` |
| Limitations, data | duplicates (33.5%, 57.5%, 21%), Response-field leakage (7 of 17), 280-alert subset | `near_duplicates.csv`, `field_lookup_predictiveness.csv`, `facts.csv`, `non_duplicate_subset.csv` |
| Limitations, tenant C later export | "we release the aggregate results": RiskFit 295 of 1,034 false escalations, CE, LightGBM, gates | `later_alerts_local_models.csv` (tenant C rows) |
| Limitations, hinge | validation folds with 2-4 malicious alerts, median Spearman -0.10 | `hinge_validation_selection.csv` |
| Limitations, hosted models | decision-model score correlation .92; Opus 16 misses, 16.5 in expectation | GUIDE: `hosted_agreement.csv`, `gate80_ties.csv` |
| Methods: Decisions deterministic across 3,300 repeat calls | | `decisions_repeatability.csv`, `facts.csv` |
| Methods (code) | serialization, RiskFit objective (Eq. 1), baselines, prompts, gates | `riskfit/serialization.py`; `riskfit/models/fastfit/modeling.py` (`lagrangian_recall_loss`, `SupConLoss`), `riskfit/models/fastfit/train.py` (`MaliciousBalancedBatchSampler`); `configs/models/`; `riskfit/prompts.py`; `riskfit/evaluation.py` (`np_threshold`, `gate`, `tau_validation`, `bootstrap_gate_counts`, `paired`, `holm`, `mcnemar_exact`, `bayes_threshold`, `reweight_factor`) |
| Evasion rules P1-P4, R0 | | `riskfit/evasion.py` |
| GUIDE setup | pre-registered selection, distinct-prompt unit, 70/15/15 split | `configs/guide_selection.yaml`, `riskfit/guide.py`, `scripts/build_guide_splits.py` |
| RQ1 "Recall targeting" | rho sweep | `TODO (pending results)`: not in this snapshot |

## Production aggregates (`aggregates/production/`)

Every file is per tenant (`A`, `B`, `C`) or pooled (`all`); "mean over folds" columns average the three
folds of a tenant (or all nine tenant-folds when pooled). Counts are malicious alerts caught or missed and
non-malicious alerts falsely escalated (`false_esc`); cost is `c * misses + false_esc`. Method names follow the
paper; `CE` is the benchmark run on a T4 GPU and `CE-A100 (lambda=0)` the paired run on RiskFit's hardware.
These numbers cannot be recomputed from this repository, because the alerts are not released; they are the
outputs of `riskfit/evaluation.py` and the recipes in `configs/models/production/` on the private folds.

| File | Contents |
|---|---|
| `data_summary.csv` | alerts per tenant and verdict, analyst corrections, agreement with the production LLM's export-time verdict, mean training-pool size and LLM malicious calls per fold |
| `default_verdicts.csv` | default (argmax / LLM) verdicts at full coverage, kappa, cost at c=10 |
| `ranking_auc.csv` | ROC- and PR-AUC of p(malicious) per tenant (fold mean) |
| `thresholds.csv` | validation-tuned thresholds (RiskFit's / Gemini's / CE's / LightGBM's default FPR, 1, 2, 5%) and the Bayes rule p >= 1/(1+c) |
| `gate80_test_quantile.csv` | 80% test-quantile gate: misses, false escalations, routed malicious, bootstrap intervals, cost at c=5/10/20, pooled Pareto set |
| `gate80_crossovers_bootstrap.csv`, `gate80_min_cost_probability.csv` | bootstrap crossovers and probability of being the cheapest method (2,000 replicates) |
| `one_to_one_vs_decisions_exemplar.csv` | each local model against the Decisions API with an exemplar: c*, interval, P(local cheaper at c=10); both gates; unperturbed and P1-P4/R0 (second bootstrap) |
| `deployable_gate.csv` | deployable gate (tau at 80% validation coverage): misses, false escalations, automated share per tenant |
| `seed_variance.csv`, `seed_deployable_cost.csv` | training seeds 42/43/44 of RiskFit, CE-A100 and MaxSim |
| `evasion.csv` | misses under P1-P4 and R0 with paired intervals, per-tenant misses, inputs changed |
| `weak_supervision.csv` | RQ3, every budget k, draw and arm, pooled; folds trained / total |
| `weak_vs_revealed_kappa_matched_folds.csv`, `weak_vs_revealed_cost_matched_folds_k25.csv` | weak labels against the revealed labels alone on the tenant-folds where both trained |
| `later_alerts_tenantA_*.csv` | the later tenant A alerts: full coverage, gates, coverage curve, Decisions API included |
| `later_alerts_local_models.csv` | later alerts of tenants A and C for RiskFit, CE and LightGBM: full coverage with Wilson intervals, gates, coverage curve (tenant C is the export excluded as a forward test) |
| `near_duplicates.csv`, `non_duplicate_subset.csv` | near-duplicate shares and the 280-alert subset without a near-duplicate |
| `field_lookup_predictiveness.csv`, `exemplar_retrieval.csv` | per-field lookup baselines (leakage audit), exemplar retrieval scores |
| `decisions_repeatability.csv`, `hinge_validation_selection.csv`, `facts.csv` | repeat calls, validation-vs-test rank correlation, scalar facts quoted in the text |

## Production data (not reproducible)

* **Schema.** Training records are JSON objects in the normalized schema of `riskfit/serialization.py`
  (`NormalizedAlert`), plus a training envelope: `alert_id`, `label` (one of `False Positive`,
  `True Positive - Benign`, `True Positive - Malicious`) and `label_source` (`gold` = the analyst corrected
  the production LLM's verdict, `silver` = the analyst confirmed it). The renderer emits 21 fields, in this
  order: alert name, severity, source, category, risk score, observables (with threat-intelligence enrichment),
  objective, description, hostname, platform, user, file name, file path, process chain, command line, detection
  engines, vendor response, trigger IOC, files accessed, MITRE tactics and techniques. The label, analyst
  comment, status and LLM verdict are never rendered.
* **Recipes.** `configs/models/production/` holds every production recipe, including the 2:1 weighting of
  analyst-corrected labels (`apply_weighting`). Put a fold in `splits/<name>/{train,val,test}.jsonl` with a
  `benchmark_split_manifest.json` (see `riskfit/guide.py::write_org`); then
  `scripts/train_guide.py --config configs/models/production/<recipe>.yaml` trains and predicts it.
* **Evaluation.** `riskfit/evaluation.py` implements every production statistic; `riskfit/evasion.py` holds the
  perturbations.

Not included, because the code is tied to private infrastructure or data: the production data export and
labelling pipeline, the Gemini 2.5 Pro client used on production alerts (its prompts are those in
`riskfit/prompts.py`), the forward-in-time, weak-supervision and evasion drivers, the near-duplicate and leakage
audits, and the cloud job launchers. Their aggregate outputs are in `aggregates/production/`.

## Tests

```bash
python tests/test_release.py --splits data/guide_splits/splits   # no training, no downloads
python tests/smoke_test.py --source-dir data/guide_raw --org 77   # trains a tiny model on one organization
```

## Notes

* GUIDE is distributed by Microsoft on Kaggle under its own terms; it is not redistributed here.
* `riskfit/models/fastfit/` is a modified copy of IBM FastFit (Apache License 2.0).
* The string `X4-GUIDE-org-` in the organization order is intentional: it is the pre-registered hash salt,
  and changing it changes the selection.

## License

Apache License 2.0 (see `LICENSE`).
