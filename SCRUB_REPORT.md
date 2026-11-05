# Scrub report

This repository was assembled from scratch for release. No history was carried over from the internal code
base, and no files were copied wholesale except the vendored, Apache-licensed FastFit trainer
(`riskfit/models/fastfit/`, comments cleaned). This report lists what was left out and how the tree was
checked. It records categories and counts only, never the matched values.

## Excluded by construction

| Category | Treatment |
|---|---|
| Production / tenant data: alerts, alert text, labels, per-alert predictions, perturbed samples | not included; only the schema (README, `riskfit/serialization.py`) and tenant-level or pooled aggregates (`aggregates/production/`) are released |
| Trained adapters, checkpoints, caches, logs, notebooks | not included; `.gitignore` keeps `data/guide_raw/`, splits, predictions, `cache/`, `results/` out of git |
| The public GUIDE dataset | not redistributed; `scripts/download_guide.sh` fetches and checksums it |
| Customer / tenant names and identifiers | removed; the code never names a tenant |
| Internal company, product, service and repository names | removed or renamed (modules rewritten under the `riskfit` package) |
| Employee names, usernames, home and scratch paths | removed; every path is a command-line argument relative to the repository |
| Cloud project ids, buckets, data-warehouse tables, container images, job ids | removed, along with the cloud launchers, data exporters, tenant registry and serving code |
| Chat / ticket references, internal commit hashes, experiment-ledger ids | removed from code and comments |
| Secrets (API keys, tokens, service-account files, `.env`) | none present. Clients read keys only from environment variables (`OPENAI_API_KEY`, `TYPESAFE_API_KEY`, `ANTHROPIC_API_KEY`). No key file or `.env` file was opened while the release was built |
| Paper author block | added (see "Release update") |

Released data: the four files in `data/guide_outputs/` were checked column by column before release. Every
prompt and exemplar id matches `guide-<OrgId>-p<16 hex>` for one of the 12 public GUIDE organizations and
appears in the public split membership. Labels and verdicts are drawn from {false_positive, benign,
malicious}. All other columns are numeric, empty, or a status from {matched, no_match}. One internal
provenance column (cache source) and one internal call index were dropped.

## Scans of the tree (initial snapshot, 47 files; see the final-paper sync below for the current tree)

Secret scanners `gitleaks` and `trufflehog` are not installed on the build machine, so a regex scan was used.

| Category | Pattern family | Hits |
|---|---|---|
| Customer / tenant names | names of the three production tenants | 0 |
| Tenant identifiers | the internal tenant hash ids | 0 |
| Internal company / product / repository names | company, product, service, internal package and repository names | 0 |
| Employee names / usernames | author and user names, the local username | 0 |
| Home and scratch paths | user-home and temporary-directory prefixes | 0 |
| Cloud storage and project ids | storage URLs, data-warehouse name, project id and number, bucket, image registry | 0 |
| Agent-tooling references | tool and worktree names, scratch directory names | 0 |
| Chat / ticket references | chat and tracker names, ticket-id shapes | 0 |
| Internal commit hashes | every commit hash cited in the internal result reports | 0 |
| Email addresses | RFC-style address pattern | 0 |
| IPv4 addresses | dotted quads | 0 |
| API keys (OpenAI/Anthropic style, AWS, GitHub, Slack) | key-prefix patterns | 0 |
| Private key blocks, bearer-token literals, JWTs, service-account JSON fields | standard shapes | 0 |
| Hard-coded key / secret / token assignments | assignment of a long literal | 0 |
| Suspicious file names | `.env`, key, secret, credential, checkpoint, pickle, notebook or log files | 0 |

Allowed hits, by design:

* **Internal experiment label inside the pre-registered hash salt** (5 occurrences in 4 files). The
  organization order is sha256(salt + OrgId) with a salt fixed before any result existed, so changing it
  would change which GUIDE organizations are selected. It identifies nothing beyond an experiment number.
* **Substring of an author surname inside scikit-learn's kappa function name** (4 occurrences, 2 files).
  The function is named after the statistician who introduced kappa, not after anyone on this project.
* **`@` characters** (matrix multiplication and decorators in Python) are not email addresses. The email
  regex reports 0.

Required plain-substring checks (case-insensitive) over every file, each with 0 hits: the tenant names, the
company and product names and their domains, the local username, the user-home and temporary-directory path
prefixes, the cloud-storage URL scheme, the data-warehouse product name, the coding-agent tool name, and the
internal repository name.

## Build checks

* `python -m compileall` over the repository: passes.
* Every module imports in an environment where the internal package is not importable. The internal package
  does not appear in `sys.modules` after the imports.
* Every file in the tree is referenced from the README.
* Equivalence with the code that produced the paper's runs:
  * The text renderer is byte-identical on 3,000 synthetic alerts covering every rendered field and
    enrichment shape.
  * The LightGBM features are identical on 7,002 records.
  * A one-epoch RiskFit training on 16 GUIDE prompts gives identical test probabilities (maximum absolute
    difference 0).
* GUIDE reproduction:
  * All 12 split fingerprints, the membership table and the tie table match.
  * All 2,158 hosted-model user messages are byte-identical to the paper's requests, including the
    five-exemplar follow-up.
  * The instruction hashes match.
  * With the paper's local predictions, `analyze_guide.py` reproduces the paper's GUIDE gate, bootstrap and
    AUC tables row for row, and every GUIDE number quoted in the paper.
* The smoke test (`tests/smoke_test.py`, one organization) passes. On the build machine LightGBM could not
  load because the OpenMP runtime is missing, so its stand-in path was exercised.

## Not included (could not be extracted cleanly without private context)

* The production data export, labelling and benchmark-split pipeline.
* The Gemini 2.5 Pro client used on production alerts. Its prompts are the ones in `riskfit/prompts.py`.
* The forward-in-time, weak-supervision, near-duplicate and leakage drivers, and the cloud job launchers.
* The production evasion driver and scorer. The perturbations themselves are in `riskfit/evasion.py`.

## Release update

* Author names and one contact email were added to `README.md` deliberately, and they are the commit author identity.
* `data/guide_predictions/` holds public GUIDE predictions only: each `alert_id` is a `guide-<org>-p<hash>` id, and the internal fields (`detected_at`, `sample_type`, `theme`, `class_logits`) and the train split were dropped.
* The license is Apache-2.0, with the standard text.


## Sync with the final paper, 2026-11-05

The previous sync targeted draft fix20. This sync targets the final paper (`paper/RiskFit_AICS2027.pdf`) and applies
the scope rule "only results that appear in the final version are in the repository": every result file was
audited against the final text, captions and tables, kept only if it backs a number, table, figure or claim there,
and mapped in the README ("Paper-to-repository map").

**Statements the final paper makes about the release.**

| Final-paper statement | Status in this tree |
|---|---|
| Code link (github.com/YanivZimmer/SocTriage) | this repository (to be pushed there) |
| "We release aggregate metrics" | `aggregates/production/` (30 tenant-level or pooled CSVs, 3 tenant x fold x seed count tables) and `aggregates/guide/` |
| "code that reproduces the GUIDE comparisons" | `scripts/analyze_guide.py` covers all 18 conditions of Table 2 and prints every GUIDE number of the final paper (`paper_numbers.txt`) |
| "with the hosted models' per-prompt GUIDE outputs" | `data/guide_outputs/`: Decisions and Jev with 0/1/5/20 exemplars, Opus 0/1, CyberSecQwen-4B and Qwen3-4B base |
| CyberSecQwen-4B "scoring details are in the code release" | `scripts/run_cybersecqwen.py` (now also documents the base-model control) |
| Tenant C's later export: "we release the aggregate results" | `aggregates/production/later_alerts_local_models.csv` (tenant C only) |
| "no alert text, customer identifiers, per-record predictions or trained adapters" | holds: the only per-record files are public GUIDE prompts (`data/guide_outputs/`, `data/guide_predictions/`) |
| "every public-data comparison is reproducible from GUIDE" | holds (see GUIDE reproduction) |

**GUIDE reproduction.** `analyze_guide.py` now loads the six local models (adds MaxSim, kNN and Tiny Decision =
Qwen3-0.6B-LoRA at 10 epochs), the 20-exemplar conditions and the Qwen3-4B base control, and writes only the
statistics the final paper reports (the deployable gate on GUIDE, test-oracle thresholds, the 1/2/5% FPR targets,
the choice-score sensitivity, the decision-model agreement table and the unreported candidate pools and paired-test
families were dropped). On the fingerprint-checked splits, every number in Table 1 (GUIDE column), Table 2 (all 18
rows: misses, false escalations, c* against RiskFit and CE, ROC-AUC mean and std, the kNN tie range), Figure 1 (left),
Figure 2 (left and caption) and the GUIDE paragraphs and Limitations is reproduced; the new rows equal the internal
analysis exactly (pooled gate counts and bootstrap intervals of all 18 conditions, all 72 local-vs-hosted crossover
rows and both candidate pools compared field by field). The 20-exemplar messages built by
`scripts/build_hosted_inputs.py --k 20` are byte-identical (sha256) to the requests for all 2,158 prompts, and the
exemplar ids match. The released local predictions were checked against the run manifests' sha256 before the
train split and internal fields (`detected_at`, `sample_type`, `theme`, `class_logits`) were dropped.

**Production aggregates.** Built from the internal per-run aggregate tables (never from per-alert files). New:
the seven-seed RiskFit vs CE-A100 comparison, the recall-target sweep, the hinge-by-sampling experiment (Table 3)
and the beta note (balanced-sampling beta=0.25 per-seed change only). Tenant x fold x seed count tables are in
`units/` (columns: tenant A/B/C, fold 0-2, seed, counts, rates; no alert, prompt, run, job or adapter identifiers);
the hyperparameter-check tables were not released because they carry internal paths, job names and adapter hashes.
Method names: the paper's "Tiny Decision" replaces the former `Qwen3-LoRA` label in every production CSV.
One labelling bug was fixed: in `later_alerts_local_models.csv` the LightGBM rows had an empty method name.

**Code.** The accumulation-window recall hinge (`recall_hinge_window: accum`, exact window gradient through a
per-micro-batch gradient buffer) and the per-epoch hinge-activity statistics used by the hinge-by-sampling runs were
ported into `riskfit/models/fastfit/modeling.py` and `train.py` (`HingeWindowMixin`, `HingeLogCallback`), with the
recipe key in `riskfit/models/fastfit_lora.py`. The default (`micro`) path is the paper's recipe and changes only
by detached statistics. `tests/test_release.py` checks on synthetic tensors that the window hinge's gradient
equals the one-graph window hinge. The beta penalty was already in the code. New recipes:
`riskfit_lambda0`, `riskfit_rho07`, `riskfit_rho095`, `hinge_by_sampling/*` (production; key-for-key equal to the
experiment configs apart from seed and description) and `maxsim_guide`, `knn_guide`, `tiny_decision_guide` (GUIDE).
`.gitignore` ignored every directory named `models/`, which would have left new files under `configs/models/` and
`riskfit/models/` untracked; it now ignores only the top-level `/models/` (and `/figures/`, `/results/`, `/out/`,
`/cache/`). `scripts/make_figures.py` now draws the final figures (9 pt text, final layout and labels).

**Keep / remove audit (result files).**

| File(s) | Decision | Final-paper place / reason |
|---|---|---|
| `data/guide_outputs/` (7 files, 2 new: 20 exemplars, Qwen3-4B base) | keep | Table 2, Figure 1 (left), GUIDE text |
| `data/guide_predictions/` (6 methods, 3 new: maxsim, knn, tiny_decision) | keep | Tables 1-2, Figure 1 (left), GUIDE text |
| `aggregates/guide/`: full_coverage, matched_fpr, auc, gate80, gate80_ties, frontier_cost, bootstrap_gate80, bootstrap_cost_crossovers, bootstrap_min_cost_probability, paired_tests, exemplar_following_test, neighbor_vote, crossovers_reweighted, summary.json, paper_numbers.txt | keep (regenerated, trimmed to reported statistics) | Tables 1-2, Figures 1-2, GUIDE paragraphs, Limitations |
| `aggregates/guide/hosted_agreement.csv` | remove | the decision-model score correlation is no longer in the paper |
| `aggregates/production/`: data_summary, default_verdicts, ranking_auc, gate80_test_quantile, one_to_one_vs_decisions_exemplar, deployable_gate, evasion, weak_supervision, weak_vs_revealed_kappa_matched_folds, weak_vs_revealed_cost_matched_folds_k25, later_alerts_tenantA_full_coverage, later_alerts_tenantA_gates, near_duplicates, non_duplicate_subset, field_lookup_predictiveness, decisions_repeatability, facts | keep | see the README map |
| `thresholds.csv` | keep, trimmed | only RiskFit's-FPR and 5%-FPR rows are reported (the production threshold table was removed) |
| `gate80_crossovers_bootstrap.csv` | keep, trimmed | only MaxSim and RiskFit rho=0.8 vs Decisions + ex. are reported |
| `gate80_min_cost_probability.csv` | keep, trimmed | only the 14-configuration pool at c=10 is reported; added the "any local" row (99.98%) |
| `later_alerts_local_models.csv` | keep, trimmed | tenant C only (Limitations); tenant A rows duplicated the tenant-A files |
| `seed_deployable_cost.csv` | keep, rebuilt | RiskFit and CE-A100 over seven seeds (the three-seed values 103 +- 23 / 106 +- 30 are superseded), MaxSim over three |
| `facts.csv` | keep | locations renamed to the final section names |
| `seed_variance.csv` | remove | three-seed hinge gain (2.3 +- 3.5) replaced by the seven-seed comparison |
| `hinge_validation_selection.csv` | remove | the hinge-selection Limitations paragraph is not in the final paper |
| `exemplar_retrieval.csv` | remove | the 66% / 38% shares are in `facts.csv`; tenants B and C are not reported |
| `later_alerts_tenantA_coverage_curve.csv` | remove | only the 80% point is reported, which is in `later_alerts_tenantA_gates.csv` |
| new: `seeds_riskfit_vs_ce_7seeds*.csv`, `units/seeds_riskfit_vs_ce_7seeds_units.csv` | add | Production, Default and tuned; Seeds, deployable gate |
| new: `recall_target_sweep_tests.csv`, `recall_target_sweep_monotonicity.csv`, `units/recall_target_sweep_units.csv` | add | Production, When the recall hinge helps (production only; the exploratory GUIDE rows are not reported) |
| new: `hinge_by_sampling_arms.csv`, `_contrasts.csv`, `_activity.csv`, `units/hinge_by_sampling_units.csv` | add | Table 3 and its paragraph; abstract and contribution (12 -> 46 +- 1); the exploratory one-seed contrast was dropped |
| new: `beta025_seed_change.csv` | add | Methods, beta note (16 +- 13) |
| not added: the beta=0.25 results beyond the per-seed change (full coverage, gates, GUIDE, evasion, forward), the natural-sampling beta experiment, the validation-tuned natural-CE analysis, the hyperparameter checks, the exploratory GUIDE recall-target rows | excluded | not in the final paper (configs and code of the two excluded experiments are not included either; only the shared beta / window-hinge code that the paper's experiments use) |

**Configs.** Every recipe in `configs/` is used by a reported result: production `riskfit`, `riskfit_rho07/08/095`,
`riskfit_lambda4`, `riskfit_lambda0`, `riskfit_beta025`, `ce`, `focal_tversky`, `maxsim`, `knn`, `setfit_lora`,
`qwen3_lora`, `hinge_by_sampling/*`; GUIDE `riskfit_guide`, `ce_guide`, `lightgbm`, `maxsim_guide`, `knn_guide`,
`tiny_decision_guide`.

**Not reproducible here / not backed by a file.**

* Production numbers cannot be recomputed (no data); they are released as aggregates.
* Compute and spend statements of the paper (about 13 minutes per encoder run, up to 1.7 A100-hours per Tiny
  Decision fold, production API calls under USD 1.50) and "on two tenant-folds, RiskFit's code with lambda=0 and
  the CE code gave identical predictions" are not backed by a released file.
* "Without changing any conclusion" (beta note) rests on the internal beta=0.25 reruns, of which only the per-seed
  false-escalation change is released, per the scope rule.
* 99.98%: the per-method shares are rounded to four decimals; the "any local" row is the unrounded share rounded
  to four decimals.
* Rounding: the paper prints "+9 false escalations (CI +4 to +15)"; the file gives +9.0 [+4.0, +14.6].
* The production best-local crossover "about 21" is attained by RiskFit rho=0.8 (21.5) at the point estimate;
  MaxSim gives 21.4. The figure labels 125 and 21 are the paper's rounded values (checked by `make_figures.py`).

**Scans of the final tree (186 files) and of the git history.** Plain-substring checks (case-insensitive) for:
the names of the three production tenants and their spelling variants (taken from an internal file and never
written down), the company cloud-project name and number, the storage URL scheme, the cloud region, the
container-registry host, the cloud-project path prefix, the internal bucket, image and code-base names,
user-home and temporary-directory prefixes, the local username, the scratch-directory names of the experiments,
internal experiment labels; regex checks for API-key prefixes (OpenAI, Google, GitHub, AWS, Slack), IPv4 addresses
and email addresses; the same checks on the paper PDF's text and on every commit's diff. All 0, except by design:
the author contact in the README (and the same address as the allowlist entry in `tests/test_release.py`); the
paper's own affiliation line ("firstname.lastname" at the company domain) and code link (the author's GitHub
account) in the PDF; the column names `ex17_*`, `ex18_*` of the 20-exemplar file (exemplar ranks); the word
"private" in "private data not released" and in the LightGBM feature for private IPv4 ranges. No file whose name
marks it as private or not for sharing was opened or copied, and no `.env`, key or credential file was read.

**Tests.** `tests/test_release.py --splits ...`: all checks pass (production column rules and final-paper spot checks,
tenant x fold x seed table rules, GUIDE snapshot numbers, regeneration of both figures, the window-hinge gradient
check, the README map covering every result file, a leak-pattern scan, and a full re-run of `analyze_guide.py`
matching the snapshot). A full 2,000-replicate re-run reproduces `aggregates/guide/` byte for byte.
`tests/smoke_test.py` trains a small model and was not run (no training was allowed); its analysis step was run on
one organization with the released predictions.
