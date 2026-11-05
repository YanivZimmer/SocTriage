# RiskFit: local triage models versus hosted decision models

Code, released per-prompt outputs and aggregate results for the paper

> **Can Small Defenders Triage Without Frontier Models? Open Local Models versus Hosted Decision Models for
> Human-on-the-Loop SOC Triage** (AICS 2027, AAAI-27 Workshop on AI for Cyber Security).
> Authors: Yaniv Zimmer, Yuval Goldberg, Noam Cohen. Contact: yaniv.zimmer@torq.io
> The final paper is in [`paper/RiskFit_AICS2027.pdf`](paper/RiskFit_AICS2027.pdf).

The paper scores SOC alert triage by unreviewed misses against false escalations, with one miss costing as much
as `c` false escalations, at default verdicts, at validation-tuned thresholds and under an 80% confidence gate.
It compares:

* **Local models trained per tenant**: RiskFit (SecureBERT 2.0 + LoRA with a hinge on soft malicious recall),
  cross-entropy (CE), focal-Tversky, MaxSim, frozen-SecureBERT kNN, LightGBM, SetFit-LoRA and the
  **tiny local decision model** ("Tiny Decision": Qwen3-0.6B fine-tuned with LoRA as a sequence classifier).
* **Hosted decision models**: the OpenAI Decisions API (`gpt-6-luna`) and, on GUIDE, TypeSafe Jev (`jev-1.14.0`),
  zero-shot and with one retrieved exemplar; on GUIDE also with 5 and 20 exemplars.
* **Generative LLMs**: Gemini 2.5 Pro on production alerts, Claude Opus 5.5 on GUIDE (zero-shot and with one
  exemplar), and CyberSecQwen-4B, an open security LLM run locally and zero-shot on GUIDE, with its base model
  Qwen3-4B-Instruct as a control.

Two datasets are used:

* **Private production alerts** (1,492 analyst-reviewed alerts from three tenants, A/B/C). The data owners
  permitted publication of aggregate results only. This repository releases those **aggregate results**
  (`aggregates/production/`: per tenant or pooled, plus tenant x fold x seed count tables for the paired seed
  tests) and the code that trains and evaluates the models on data in the same schema, but no alert text, labels,
  per-alert predictions, customer identifiers or trained adapters.
* **The public GUIDE benchmark** (12 organizations). Every GUIDE number in the paper is reproducible from this
  repository: the dataset download, the split builder with fingerprint checks, the training and analysis code,
  the hosted and open models' per-prompt outputs and the paper's local-model per-prompt predictions.

Scope rule: the repository holds only results that appear in the final paper. Every result file is mapped to the
place in the paper that uses it (section "Paper-to-repository map"); `SCRUB_REPORT.md` lists what was removed.

## What is and is not released

| Released | Not released |
|---|---|
| All code for the GUIDE comparisons (splits, local models, hosted-model clients, open-LLM scoring, analysis, figures) | Production alerts, alert text, raw alerts |
| Per-prompt GUIDE outputs of every hosted / open-LLM condition in the paper: Decisions and Jev with 0 / 1 / 5 / 20 exemplars, Opus with 0 / 1, CyberSecQwen-4B and Qwen3-4B base zero-shot | Per-alert production labels, predictions or confidences |
| The paper's per-prompt GUIDE predictions of the six local models (RiskFit, CE, LightGBM, MaxSim, kNN, Tiny Decision) | Trained adapters and checkpoints |
| Aggregate production metrics behind every production table, figure and text number (per tenant or pooled; tenant x fold x seed counts for the seed and hinge tests) | Customer names, hosts, users, IPs, internal identifiers |
| Training recipes for every production model and experiment arm in the paper, the evaluation code and the evasion perturbations | The production export / labelling pipeline, the Gemini client, the cloud launchers |
| A snapshot of the GUIDE analysis output (`aggregates/guide/`) and the final paper PDF | The GUIDE dataset itself (download script provided) |

## Repository layout

```
paper/RiskFit_AICS2027.pdf    the final paper
configs/
  guide_selection.yaml        pre-registered GUIDE organization selection and split rule, expected checksums
  guide_orgs.csv              the 12 organizations: counts and split fingerprints
  models/                     GUIDE recipes: riskfit_guide, ce_guide, lightgbm, maxsim_guide, knn_guide,
                              tiny_decision_guide (.yaml)
  models/production/          production recipes: riskfit, riskfit_rho07, riskfit_rho08, riskfit_rho095,
                              riskfit_lambda4, riskfit_lambda0 (CE-A100), riskfit_beta025 (beta note), ce,
                              focal_tversky, maxsim, knn, setfit_lora, qwen3_lora (Tiny Decision) (.yaml)
  models/production/hinge_by_sampling/
                              the Table 3 arms not covered above: balanced_rho095_lambda5, balanced_rho099_lambda8,
                              natural_ce, natural_rho09_lambda2, natural_rho095_lambda5, natural_rho099_lambda8
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
  models/fastfit/             vendored FastFit (__init__.py, modeling.py, train.py) with the RiskFit objective,
                              the optional beta penalty, the accumulation-window hinge, hinge-activity logging and
                              the balanced batch sampler
  models/tabular.py           LightGBM and its structured features
  models/baselines.py         frozen-SecureBERT kNN, SetFit-LoRA, Qwen3-0.6B-LoRA (Tiny Decision)
scripts/
  download_guide.sh           download and checksum the public GUIDE dataset (not redistributed here)
  build_guide_splits.py       rebuild the 12-organization benchmark and verify fingerprints
  train_guide.py              train a local model per organization, write per-prompt predictions
  build_hosted_inputs.py      zero-shot and 1-, 5- or 20-exemplar messages for the hosted models (no API call)
  run_decisions.py            OpenAI Decisions API client
  run_jev.py                  TypeSafe Jev client
  run_opus.py                 Claude Opus 5.5 client (Messages API, or the Claude Code CLI path used in the paper)
  run_cybersecqwen.py         CyberSecQwen-4B / Qwen3-4B base zero-shot label scoring (local; --check-tokens needs no weights)
  analyze_guide.py            every GUIDE statistic in the paper -> paper_numbers.txt
  make_figures.py             Figures 1 and 2 from released CSVs
data/guide_outputs/           released per-prompt outputs of the hosted and open models on GUIDE
data/guide_predictions/       the paper's per-prompt GUIDE predictions of the six local models
aggregates/production/        aggregate production results (tenant A/B/C or pooled), one CSV per table/analysis
aggregates/production/units/  tenant x fold x seed counts behind the paired seed and hinge tests
aggregates/guide/             snapshot of analyze_guide.py's output on the released predictions (18 conditions)
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
`cisco-ai/SecureBERT2.0-biencoder` (about 600 MB) from the Hugging Face Hub on first use. SetFit-LoRA downloads
`sentence-transformers/paraphrase-mpnet-base-v2` and Tiny Decision `Qwen/Qwen3-0.6B-Base`. Re-scoring the open
LLMs needs `athena129/CyberSecQwen-4B` and `Qwen/Qwen3-4B-Instruct-2507` (about 8 GB each, bf16).

| Step | Hardware | Time |
|---|---|---|
| Build the 12 GUIDE splits | laptop CPU | about 1 minute (reads 3.5 GB of CSV) |
| Train RiskFit or CE, 12 organizations | one NVIDIA A100 40 GB | about 3 GPU-hours per method (20-epoch ceiling, early stopping) |
| Train MaxSim and Tiny Decision (10 epochs), 12 organizations | one NVIDIA A100 40 GB | 6.75 A100-hours for both together in the paper's runs |
| kNN, 12 organizations | CPU | about 3 CPU node-hours in the paper's runs (embeddings only) |
| Train LightGBM, 12 organizations | 8 CPU cores | about 15 minutes |
| Analysis (18 conditions, 2,000 bootstrap replicates) | laptop CPU | under 1 minute |
| Figures from released CSVs | laptop CPU | seconds |
| Score an open LLM (optional) | Apple M-series GPU (MPS) or one GPU | about 41 minutes for CyberSecQwen-4B (1,843 distinct prompts) |
| Re-query hosted models (optional) | network | Decisions and Jev about USD 1 for 0/1/5 exemplars and USD 1.33 for 20; Opus about 3.8M input tokens |

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
for m in riskfit ce lightgbm maxsim knn tiny_decision; do
  python scripts/train_guide.py --splits data/guide_splits/splits --method $m --out data/predictions
done

# 4. Hosted-model inputs: byte-identical to the paper's requests (instruction hashes, exemplar ids, and for k=5
#    and k=20 the sha256 of every multi-exemplar message are checked against the released outputs)
python scripts/build_hosted_inputs.py --splits data/guide_splits/splits --out data/hosted_inputs.jsonl
python scripts/build_hosted_inputs.py --splits data/guide_splits/splits --k 5 --out data/hosted_inputs_k5.jsonl
python scripts/build_hosted_inputs.py --splits data/guide_splits/splits --k 20 --out data/hosted_inputs_k20.jsonl

# 5. All GUIDE statistics (18 conditions), using the released outputs in data/guide_outputs/
python scripts/analyze_guide.py --predictions data/guide_predictions --splits data/guide_splits/splits --out results/guide
cat results/guide/paper_numbers.txt          # every GUIDE number in the paper, grouped by table / figure / paragraph
diff -r results/guide aggregates/guide        # identical to the released snapshot

# 6. Figures 1 and 2 (GUIDE panels from step 5 or from the snapshot; production panels from aggregates/production)
python scripts/make_figures.py --guide results/guide --out figures
```

Step 5 needs no API access. With the paper's local-model predictions (`data/guide_predictions/`) it reproduces
every GUIDE number in the paper exactly, including bootstrap intervals (the bootstrap seed and resampling order are
fixed, and the resampling does not depend on which methods are analyzed). Retraining the local models (step 3) on
other hardware or library builds can move individual predictions slightly, so expect small differences in the
local models' counts. The paper's GUIDE kNN ran on CPU (embeddings on the A100 build returned NaN), and its
confidences tie heavily at the gate cutoff (misses 19 to 21 with tie order; `gate80_ties.csv`).

**Re-querying the hosted models (optional).** The released outputs are what the paper analyzes. Keys are read only
from the environment:

```bash
OPENAI_API_KEY=...    python scripts/run_decisions.py --inputs data/hosted_inputs.jsonl --out out/decisions_guide_per_prompt.csv
TYPESAFE_API_KEY=...  python scripts/run_jev.py       --inputs data/hosted_inputs.jsonl --out out/jev_guide_per_prompt.csv
ANTHROPIC_API_KEY=... python scripts/run_opus.py      --inputs data/hosted_inputs.jsonl --out out/opus_guide_per_prompt.csv
python scripts/run_opus.py --via cli ...   # the Claude Code CLI path used in the paper (no temperature control)
# 5 or 20 exemplars: the same clients on data/hosted_inputs_k5.jsonl / data/hosted_inputs_k20.jsonl (their ex_*
# columns are then the multi-exemplar condition); pass the results to analyze_guide.py with --five-exemplar /
# --twenty-exemplar in the released column layout
python scripts/analyze_guide.py --hosted out --predictions data/guide_predictions --splits data/guide_splits/splits --out results/guide_requery
```

Decisions (`gpt-6-luna`, queried 2026-10-29 to 2026-11-05) and Jev (`jev-1.14.0`, 2026-11-01 to 2026-11-05) have
no dated snapshots, and Claude Opus 5.5 (2026-11-03, medium effort) was run once and is not deterministic, so a
re-query can differ from the released outputs. The Opus client does not enable server-side refusal fallbacks;
a refusal is recorded as unresolved.

**Open-LLM scoring.** `scripts/run_cybersecqwen.py` documents and implements the scoring used for the paper (the
paper's "scoring details are in the code release"): `athena129/CyberSecQwen-4B` at revision `6c82bf13`, one user
turn in the model's chat template holding the decision models' instructions, the byte-identical zero-shot alert
message and a one-word question; each of the three verdict labels is teacher-forced and its token log-probabilities
summed; `p(malicious)` is the softmax over the three sums, the verdict is the argmax and the gate confidence is the
largest probability. No sampling. The base-model control uses `Qwen/Qwen3-4B-Instruct-2507` at revision `cdbee75f`
with the same prompt and scoring. `--check-tokens` re-renders all 2,158 prompts and compares their token counts with
the released file using only the tokenizer.

### Released per-prompt outputs (`data/guide_outputs/`)

Public GUIDE identifiers and model outputs only. Prompt ids are `guide-<OrgId>-p<sha256(prompt text)[:16]>`;
labels are prompt majority labels (`false_positive`, `benign`, `malicious`).

| File | Rows | Paper conditions (Table 2, Figure 1) | Columns |
|---|---|---|---|
| `decisions_guide_per_prompt.csv` | 2,158 (1,079 test + 1,079 validation) | Decisions zero-shot, + 1 exemplar | exemplar id/status/score; for `ex` and `zs`: verdict, confidence, choice probabilities (`p_fp`, `p_benign`, `p_mal_q1`), predicate probability (`p_mal_q2`) |
| `jev_guide_per_prompt.csv` | 2,158 | Jev zero-shot, + 1 exemplar | as above; `p_mal_q2` is the yes/no probability, plus input token counts |
| `decisions_jev_5ex_guide_per_prompt.csv` | 2,158 | Decisions + 5 exemplars, Jev + 5 exemplars | the five exemplar ids, scores and labels, tie counts, the sha256 of each five-exemplar message, `dec5_*` and `jev5_*` outputs |
| `decisions_jev_20ex_guide_per_prompt.csv` | 2,158 | Decisions + 20 exemplars, Jev + 20 exemplars | the twenty exemplar ids, scores and labels, tie counts at the 20th, the sha256 of each twenty-exemplar message, `dec20_*` and `jev20_*` outputs |
| `opus_guide_per_prompt.csv` | 2,158 (1,079 test prompts x 2 conditions) | Opus zero-shot, + 1 exemplar | verdict, verbalized confidence, `p_malicious` |
| `cybersecqwen4b_zeroshot_guide_per_prompt.csv` | 2,158 | CyberSecQwen-4B zero-shot | verdict, confidence, the three label probabilities and summed log-probabilities, first-token label mass, prompt tokens |
| `qwen3_4b_base_zeroshot_guide_per_prompt.csv` | 2,158 | Qwen3-4B base zero-shot | same columns as the CyberSecQwen-4B file |

`data/guide_predictions/{riskfit,ce,lightgbm,maxsim,knn,tiny_decision}/org_<id>.jsonl` hold the paper's
per-prompt GUIDE test and validation predictions (2,158 rows per method; `alert_id`, `split`, `expected_label`,
`predicted_label`, `confidence`, `class_probs`).

## Paper-to-repository map

Numbering follows the final paper: Tables 1-6 and Figures 1-2. "GUIDE" files are outputs of
`scripts/analyze_guide.py`, released in `aggregates/guide/` (`paper_numbers.txt` prints each GUIDE number under the
table or paragraph that quotes it). "Production" files are in `aggregates/production/`. In the CSVs, the paper's
"Tiny Decision" is `Tiny Decision` (Qwen3-0.6B-LoRA), "CE-A100" is `CE-A100 (lambda=0)` and "+ ex." is
`+ exemplar`.

**Tables and figures**

| Paper element | What | Repository file(s) |
|---|---|---|
| Table 1, GUIDE column | gate 156 / 185, Tiny Decision 150, Decisions + ex. 500, + 20 ex. 210; full coverage 329 / 740; c* 125 / 55 | GUIDE: `frontier_cost.csv`, `full_coverage.csv`, `bootstrap_cost_crossovers.csv` |
| Table 1, production column | gate 118 / 98, Tiny Decision 105, Decisions + ex. 154; full coverage 234 / 233; c* 14 / 16 | `gate80_test_quantile.csv`, `default_verdicts.csv`, `one_to_one_vs_decisions_exemplar.csv` |
| Table 2 (GUIDE gate, 18 conditions) | misses / FE, c* vs RiskFit and CE, ROC-AUC mean +- std; caption: kNN 19-21 with tie order | GUIDE: `frontier_cost.csv`, `gate80.csv` (per organization), `bootstrap_cost_crossovers.csv`, `auc.csv`, `gate80_ties.csv`; inputs `data/guide_outputs/`, `data/guide_predictions/` |
| Table 3 (recall hinge by sampling) | caught, FE, gate cost, mean +- std over seeds 0-2 | `hinge_by_sampling_arms.csv`; per tenant-fold-seed counts `units/hinge_by_sampling_units.csv`; recipes `configs/models/production/riskfit_lambda0.yaml` (balanced CE), `riskfit.yaml` (balanced rho .9 / lambda 2), `hinge_by_sampling/*.yaml` |
| Table 4 (production gate) | misses / FE pooled, the four Pareto-optimal configurations, omitted zero-shot rows | `gate80_test_quantile.csv` (`pareto_pooled`) |
| Table 5 (evasion) | misses under P1-P4 and R0, bold = paired interval excludes zero | `evasion.csv`; perturbations `riskfit/evasion.py` |
| Table 6 (weak supervision at 25%) | caught / FE / kappa, footnotes on failed folds | `weak_supervision.csv` |
| Figure 1, left | GUIDE default verdicts, 17 conditions | GUIDE: `full_coverage.csv`; `scripts/make_figures.py` |
| Figure 1, right + caption | production default verdicts, 13 methods; CE-A100 50 with 28 | `default_verdicts.csv`; `scripts/make_figures.py` |
| Figure 2 + caption | cost curves; production crossovers 14 / 16 / 11 and 115 / 117 / 78; GUIDE best-local crossover 125; reweighted 896 / 695 | GUIDE: `frontier_cost.csv`, `crossovers_reweighted.csv`; production: `gate80_test_quantile.csv`, `one_to_one_vs_decisions_exemplar.csv`, `gate80_crossovers_bootstrap.csv`; `scripts/make_figures.py` |

**Text**

| Paper place | Claim / numbers | Repository file(s) |
|---|---|---|
| Abstract, contributions | crossovers 14 and 125; Tiny Decision cheaper than Decisions + ex. on both datasets (105 vs 154; 150 vs 500); recall hinge 12 -> 46 +- 1 of 73 under natural sampling (Holm p < 0.001, every seed); weak-label break-even about 20 (CE) / 27 (RiskFit) | as Tables 1-3; `hinge_by_sampling_contrasts.csv`; `weak_vs_revealed_cost_matched_folds_k25.csv` (`breakeven_c_weak_vs_human_only`) |
| Methods, RiskFit objective | Eq. 1; balanced 50/50 micro-batches of four, accumulation over nine; rho / lambda / gamma; CE-A100 | `riskfit/models/fastfit/modeling.py` (`lagrangian_recall_loss`, `SupConLoss`), `riskfit/models/fastfit/train.py` (`MaliciousBalancedBatchSampler`); `configs/models/production/riskfit*.yaml` |
| Methods, beta note | beta = 0.25 with balanced sampling cut false escalations by 16 +- 13 per seed (seeds 42-44) | `beta025_seed_change.csv` (scope `all`: `fe_mean_sd`); recipe `configs/models/production/riskfit_beta025.yaml` |
| Methods, Baselines | Tiny Decision (Qwen3-0.6B-LoRA with a score head); kNN k=5; label-blind exemplar retrieval; Decisions deterministic across 3,300 repeated calls; Jev / Opus / CyberSecQwen-4B and its base model | `riskfit/models/baselines.py` (`QwenLoRA`, `KNNBaseline`), `configs/models/production/qwen3_lora.yaml`, `configs/models/tiny_decision_guide.yaml`; `riskfit/prompts.py`; `decisions_repeatability.csv`, `facts.csv`; `scripts/run_*.py` |
| Methods, operating points and gate | test-quantile and deployable gates, validation-tuned thresholds | `riskfit/evaluation.py` (`gate`, `tau_validation`, `np_threshold`, `bootstrap_gate_counts`, `paired`, `holm`) |
| Setup, GUIDE | pre-registered selection, distinct-prompt unit, 70/15/15 split, 1,079 test prompts (456 malicious); 5 and 20 exemplars | `configs/guide_selection.yaml`, `riskfit/guide.py`, `scripts/build_guide_splits.py`, `scripts/build_hosted_inputs.py` |
| Setup, production alerts | 412 / 707 / 373 alerts, 17 / 39 / 17 malicious; 528 corrections (6 malicious); 964 uncorrected, 943 equal to the export-time verdict | `data_summary.csv` |
| Setup, recall-hinge experiments | seven-seed RiskFit vs CE-A100; rho in {0.7, 0.8, 0.9, 0.95} x 3 seeds; sampling x loss over seeds 0-2; accumulation-window hinge over 36 alerts | configs as above (`riskfit_rho07/08/095.yaml`, `hinge_by_sampling/`), `recall_hinge_window` in `riskfit/models/fastfit/modeling.py` (`_window_hinge`) and `train.py` (`HingeWindowMixin`) |
| GUIDE, Confidence gating | Tiny Decision lowest at c=5 and 10, RiskFit at c=20; cheapest in 64 / 30 / 6% without the 20-exemplar conditions; Holm p about 1 Tiny Decision vs RiskFit; 97.6% / 86.8% with all 18; RiskFit / CE dominate Opus + ex. in 84 / 60%; 8 vs 11 misses (3-15, 5-19), nine times the FE, Holm p = 0.003; c* 125 (30 to unbounded); 20 exemplars 12 / 90 (210), dominated in 58 / 52%, CE c* 18; CyberSecQwen c* 41; base ROC-AUC .45 | GUIDE: `frontier_cost.csv`, `bootstrap_min_cost_probability.csv`, `paired_tests.csv`, `summary.json`, `bootstrap_gate80.csv`, `bootstrap_cost_crossovers.csv` (`boot_share_local_cheaper_at_all_c`), `auc.csv` |
| GUIDE, Default and tuned operating points | default verdicts of the six local models; hosted 399-452 caught with 126-620 FE; CyberSecQwen 1,077 of 1,079 malicious; RiskFit vs LightGBM recall +0.15 (Holm p = 0.024), vs CE +0.01; tuned at RiskFit's FPR: Holm p = 0.008, costs 1,381 / 1,173 (5 ex.), 541 / 875 (20 ex.), 325 / 380 / 451; ROC-AUC ordering | GUIDE: `full_coverage.csv`, `paired_tests.csv`, `matched_fpr.csv` (`cost_c10`), `auc.csv` |
| GUIDE, Exemplars | exemplar label correct 86%; following 55 / 51% (45 / 43% zero-shot), Opus 79% (45%); 85 / 68% with five, 89 / 71% with twenty; FE -79% / -44% from one exemplar; neighbor vote 410 / 75, cost 535 vs 329 / 340 | GUIDE: `exemplar_following_test.csv`, `frontier_cost.csv`, `neighbor_vote.csv`, `full_coverage.csv` |
| Production, Default and tuned | seven seeds: RiskFit 50.0 +- 3.2 caught, 39 +- 13 FE; CE-A100 49.4 +- 2.4, 30 +- 7; +0.6 (CI -1.1 to +2.3), +9 (CI +4 to +15); equal cost under every gate (Holm p = 1.0); seed 42 most favorable; Decisions 71 / 70 with 213 / 1,170; ROC-AUC per tenant; kNN 55 / 48, RiskFit 56 / 93, Decisions + ex. 54 / 73 at RiskFit's FPR | `seeds_riskfit_vs_ce_7seeds.csv`, `seeds_riskfit_vs_ce_7seeds_tests.csv`, `units/seeds_riskfit_vs_ce_7seeds_units.csv`, `default_verdicts.csv`, `ranking_auc.csv`, `thresholds.csv` |
| Production, When the recall hinge helps | rho 0.7 / 0.8 / 0.9 / 0.95 -> recall 0.72 / 0.74 / 0.73 / 0.74 (Spearman 0.008); no closer to target than CE thresholded on validation (Holm p = 1.0); balanced: soft recall 0.36 -> 0.99, hinge active on every first-epoch step and on 3 / 15 / 5% by the last, no difference from CE (Holm p >= 0.17); natural: CE 11.7 caught, soft recall 0.46, hinges 45-47 (Holm p < 0.001, every seed), about 28 more FE; interaction Holm p < 0.001 (catches, full cost) and <= 0.032 (gate cost) | `recall_target_sweep_monotonicity.csv`, `recall_target_sweep_tests.csv`, `units/recall_target_sweep_units.csv`; `hinge_by_sampling_activity.csv`, `hinge_by_sampling_contrasts.csv`, `hinge_by_sampling_arms.csv` |
| Production, Confidence gating | Pareto set; costs 74 / 85 / 98 / 118 vs 151 / 154; a local model cheapest in 99.98% of replicates; MaxSim / rho=0.8 intervals 2-14 / 2-13; one-to-one 89 / 95.5 / 93 / 99 / 42 / 67% | `gate80_test_quantile.csv`, `gate80_min_cost_probability.csv` (row `any of the 10 local configurations`), `one_to_one_vs_decisions_exemplar.csv` |
| Production, Seeds, deployable gate and crossover | deployable cost 104 +- 27 (RiskFit, 7 seeds), 101 +- 20 (CE-A100, 7 seeds), 90 +- 46 (MaxSim, 3 seeds) vs 155, every seed below; full coverage 233 vs 234; kNN at 5% FPR 204; RiskFit deployable 6 / 26, cheaper in 98%, automating 68 / 75 / 82%; c* 14 / 16; about 21 (rho=0.8 21.5, MaxSim 21.4, 11-71); Gemini + ex. 11 / 41, at most six confidence values per fold, tenant B 9 / 36 | `seed_deployable_cost.csv`, `default_verdicts.csv`, `thresholds.csv`, `deployable_gate.csv`, `one_to_one_vs_decisions_exemplar.csv`, `gate80_crossovers_bootstrap.csv`, `facts.csv`, `gate80_test_quantile.csv` (per tenant) |
| Production, Later alerts (tenant A) | 433 alerts (11 malicious); full coverage 9 / 8, 9 / 50, 5 / 2; gate 2 / 1, 2 / 0, 2 / 28, 1 / 56, c* about 55; benchmark comparison at the gate; FE rate 0.045 -> 0.18 (RiskFit 0.019); 66% vs 38% without an exemplar; carried-forward automation 15 / 20 / 39 / 72 / 93% (Decisions + ex. 0 / 20) | `later_alerts_tenantA_full_coverage.csv`, `later_alerts_tenantA_gates.csv`, `gate80_test_quantile.csv` (tenant A), `facts.csv` |
| Production, Evasive alerts | inputs changed 29 / 24 / 30 / 47; encoders at most +2; Pareto set and c* about 21 unchanged; LightGBM +10 (4-17); R0: 7 of tenant A's 17 | `evasion.csv`, `one_to_one_vs_decisions_exemplar.csv` (perturbed conditions), `facts.csv` |
| Production, Weak supervision | kappa .27 / .30 vs .59 / .60; weak labels win on kappa at k <= 5%, revealed labels at 25-50%; RiskFit at 25% on matched folds 57 of 67 with 546 vs 39 with 55; tenant B about 330 of 471 vs 26 | `weak_supervision.csv`, `weak_vs_revealed_kappa_matched_folds.csv`, `weak_vs_revealed_cost_matched_folds_k25.csv`, `data_summary.csv`, `facts.csv` |
| Limitations, Data and evaluation | 33.5% exact duplicates (A), 57.5% / 21% at cosine >= 0.99; Response field and enrichment may carry outcome information; tenant C later export, RiskFit 295 of 1,034 false escalations, "we release the aggregate results"; 280-alert subset | `near_duplicates.csv`, `field_lookup_predictiveness.csv`, `later_alerts_local_models.csv` (tenant C: full coverage, validation-threshold gates, coverage curve), `non_duplicate_subset.csv` |
| Limitations, Where hosted models are competitive | Decisions + ex. ranks tenants A and B better than every local model; Opus ties: 16 misses, 16.5 expected; 20 exemplars 210 vs 156 | `ranking_auc.csv`; GUIDE: `gate80_ties.csv`, `frontier_cost.csv` |
| Limitations, Adversaries and scope | the alert name still carries the indicator for 15 of the 30 P3-changed alerts | `facts.csv` |
| Ethical Statement | aggregate metrics, GUIDE code and per-prompt GUIDE outputs released; no alert text, customer identifiers, per-record predictions or trained adapters | this repository; `SCRUB_REPORT.md` |

## Production aggregates (`aggregates/production/`)

Every top-level file is per tenant (`A`, `B`, `C`) or pooled (`all`); "mean over folds" columns average the three
folds of a tenant (or all nine tenant-folds when pooled). Counts are malicious alerts caught or missed and
non-malicious alerts falsely escalated (`false_esc`, `fe`); cost is `c * misses + false_esc`. Method names follow
the paper; `CE` is the benchmark run on a T4 GPU and `CE-A100 (lambda=0)` the paired run on RiskFit's hardware.
`units/` holds tenant x fold x seed counts (no alert, prompt or run identifiers), the units of the paired
Wilcoxon tests. These numbers cannot be recomputed from this repository, because the alerts are not released; they
are the outputs of `riskfit/evaluation.py` and the recipes in `configs/models/production/` on the private folds.

| File | Contents |
|---|---|
| `data_summary.csv` | alerts per tenant and verdict, analyst corrections, agreement with the production LLM's export-time verdict, mean training-pool size and LLM malicious calls per fold |
| `default_verdicts.csv` | default (argmax / LLM) verdicts at full coverage, kappa, cost at c=10 |
| `ranking_auc.csv` | ROC- and PR-AUC of p(malicious) per tenant (fold mean) |
| `thresholds.csv` | validation-tuned thresholds at RiskFit's default FPR and at 5% FPR |
| `gate80_test_quantile.csv` | 80% test-quantile gate: misses, false escalations, routed malicious, bootstrap intervals, cost at c=5/10/20, pooled Pareto set |
| `gate80_crossovers_bootstrap.csv` | bootstrap crossovers of MaxSim and RiskFit rho=0.8 with the Decisions API + exemplar |
| `gate80_min_cost_probability.csv` | probability of being the cheapest of the 14 configurations at c=10 (2,000 replicates), and of any local one |
| `one_to_one_vs_decisions_exemplar.csv` | each evasion-study local model against the Decisions API with an exemplar: c*, interval, P(local cheaper at c=10); both gates; unperturbed and P1-P4/R0 |
| `deployable_gate.csv` | deployable gate (tau at 80% validation coverage): misses, false escalations, automated share per tenant |
| `seed_deployable_cost.csv` | deployable-gate cost at c=10 per seed: RiskFit and CE-A100 (seeds 42-48), MaxSim (42-44) |
| `seeds_riskfit_vs_ce_7seeds.csv`, `seeds_riskfit_vs_ce_7seeds_tests.csv` | seven seeds of RiskFit and CE-A100: per-seed totals; differences with bootstrap CIs, Wilcoxon over 63 units, Holm |
| `recall_target_sweep_tests.csv`, `recall_target_sweep_monotonicity.csv` | RiskFit trained with rho in {0.7, 0.8, 0.9, 0.95}: distance to target against CE (default and validation-thresholded), Holm; realized recall per rho and Spearman |
| `hinge_by_sampling_arms.csv`, `hinge_by_sampling_contrasts.csv`, `hinge_by_sampling_activity.csv` | Table 3 arms; the nine pre-registered contrasts (hinge - CE per sampling, interactions) per metric with CIs and Holm; hinge activity and training soft recall by epoch |
| `beta025_seed_change.csv` | RiskFit beta=0.25 minus beta=0 with balanced sampling, per tenant and pooled, seeds 42-44 |
| `evasion.csv` | misses under P1-P4 and R0 with paired intervals, per-tenant misses, inputs changed |
| `weak_supervision.csv` | weak supervision, every budget k, draw and arm, pooled; folds trained / total |
| `weak_vs_revealed_kappa_matched_folds.csv`, `weak_vs_revealed_cost_matched_folds_k25.csv` | weak labels against the revealed labels alone on the tenant-folds where both trained |
| `later_alerts_tenantA_full_coverage.csv`, `later_alerts_tenantA_gates.csv` | the later tenant A alerts: full coverage, gates (test-quantile and carried-forward validation threshold), Decisions API included |
| `later_alerts_local_models.csv` | tenant C's later export (not used as a forward test): RiskFit, CE and LightGBM at full coverage with Wilson intervals, validation-threshold gates, coverage curve |
| `near_duplicates.csv`, `non_duplicate_subset.csv` | near-duplicate shares and the 280-alert subset without a near-duplicate |
| `field_lookup_predictiveness.csv` | per-field lookup baselines (leakage audit behind the Response-field and enrichment caveat) |
| `decisions_repeatability.csv`, `facts.csv` | repeat calls of the Decisions API; scalar facts quoted in the text |
| `units/seeds_riskfit_vs_ce_7seeds_units.csv`, `units/recall_target_sweep_units.csv`, `units/hinge_by_sampling_units.csv` | tenant x fold x seed counts behind the three experiments above |

## Production data (not reproducible)

* **Schema.** Training records are JSON objects in the normalized schema of `riskfit/serialization.py`
  (`NormalizedAlert`), plus a training envelope: `alert_id`, `label` (one of `False Positive`,
  `True Positive - Benign`, `True Positive - Malicious`) and `label_source` (`gold` = the analyst corrected
  the production LLM's verdict, `silver` = the analyst confirmed it). The renderer emits 21 fields, in this
  order: alert name, severity, source, category, risk score, observables (with threat-intelligence enrichment),
  objective, description, hostname, platform, user, file name, file path, process chain, command line, detection
  engines, vendor response, trigger IOC, files accessed, MITRE tactics and techniques. The label, analyst
  comment, status and LLM verdict are never rendered.
* **Recipes.** `configs/models/production/` holds every production recipe in the paper, including the 2:1 weighting
  of analyst-corrected labels (`apply_weighting`). Put a fold in `splits/<name>/{train,val,test}.jsonl` with a
  `benchmark_split_manifest.json` (see `riskfit/guide.py::write_org`); then
  `scripts/train_guide.py --config configs/models/production/<recipe>.yaml --seed <seed>` trains and predicts it.
  Seeds: 42 for the benchmark runs; 42-48 for the seven-seed comparison (`riskfit.yaml`, `riskfit_lambda0.yaml`);
  42-44 for the rho sweep and the beta note; 0-2 for the hinge-by-sampling arms.
* **Hinge options.** `recall_hinge_window: micro` (default; the paper's recipe) computes soft recall per
  micro-batch; `accum` computes it over the gradient-accumulation window (9 micro-batches of 4 = 36 alerts) with an
  exact window gradient, as in the natural-sampling arms. `balance_malicious_batches: false` gives natural sampling.
  Per-epoch hinge activity is printed (`HINGE_EPOCH`) and written to `$RISKFIT_HINGE_LOG` if set.
  `false_positive_penalty_beta` is the soft false-positive penalty (0 throughout; 0.25 in the beta note).
* **Evaluation.** `riskfit/evaluation.py` implements every production statistic; `riskfit/evasion.py` holds the
  perturbations.

Not included, because the code is tied to private infrastructure or data: the production data export and
labelling pipeline, the Gemini 2.5 Pro client used on production alerts (its prompts are those in
`riskfit/prompts.py`), the forward-in-time, weak-supervision and evasion drivers, the near-duplicate and leakage
audits, the cloud job launchers and the per-run aggregation scripts. Their aggregate outputs are in
`aggregates/production/`.

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
