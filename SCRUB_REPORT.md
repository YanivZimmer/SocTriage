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

## Scans of the final tree (47 files)

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


## Sync with the paper draft (fix20), 2026-11-05

Checked every statement the draft makes about the release, and every GUIDE number in it, against this tree.

**Statements about the release (Ethical Statement, Methods, Limitations).**

| Draft statement | Status in this tree |
|---|---|
| "We release aggregate metrics" | `aggregates/production/` (26 CSVs, tenant A/B/C or pooled only) and `aggregates/guide/` |
| "code that reproduces the GUIDE comparisons" | `scripts/analyze_guide.py` now covers all 12 GUIDE methods, including Decisions and Jev with five exemplars and CyberSecQwen-4B, and prints every GUIDE number of the draft (`paper_numbers.txt`) |
| "the hosted models' per-prompt GUIDE outputs" | `data/guide_outputs/`: Decisions 0/1/5 exemplars, Jev 0/1/5, Opus 0/1, CyberSecQwen-4B |
| CyberSecQwen-4B "scoring details are in the code release" | `scripts/run_cybersecqwen.py` (new) and `riskfit.prompts.open_llm_user_prompt` (new) |
| Tenant C's later alerts: "we release the aggregate results" | `aggregates/production/later_alerts_local_models.csv` (tenant C rows: counts, rates with Wilson intervals, kappa, gates, coverage curve) |
| "no alert text, customer identifiers, per-record predictions or trained adapters" | holds: the only per-record files are public GUIDE prompts (`data/guide_outputs/`, `data/guide_predictions/`) |

**GUIDE reproduction.** On a fresh split rebuild from the checksummed archive (ALL CHECKS PASSED), the analysis on
the released predictions and outputs reproduces every GUIDE number in the draft: Table 1 (GUIDE column), Table 3
(all seven rows and the bold cells), Table 4 (misses, false escalations, c* against RiskFit and CE, ROC-AUC mean
and standard deviation), Figure 2 (left), Figure 3 (left) and its caption, and the GUIDE numbers in RQ1, RQ2,
"Exemplars" and Limitations. Adding the five-exemplar and CyberSecQwen conditions leaves every row of the nine
original methods byte-identical (checked file by file), apart from three `dominated_by` labels in the frontier
table that now also name the new conditions. The new rows equal the internal follow-up analyses exactly
(gate counts, bootstrap intervals, crossovers, Holm-adjusted tests). The CyberSecQwen prompt built by the new code is
byte-identical to the scored prompt for all 2,158 prompts, and its token counts equal the released ones.

**Production aggregates.** Built from the internal per-run aggregate tables (never from per-alert files), by a
build script that is not released because it reads internal paths. Checks on the result:

* Granularity: every row is tenant A, B, C or pooled; no fold-level metric columns, run or job ids, image
  digests, pipeline names, alert or prompt ids, or free-text notes. Fold-level rows of the sources were averaged
  or summed; internal notes columns were dropped. `tests/test_release.py` enforces the column rules.
* Values: the build asserts, and the test re-checks, the draft's Table 2, Table 5 (all 14 rows and the Pareto
  set), Table 6, Table 7 and the Figure 2 (right) bars; the remaining production numbers of the text were
  compared by hand (see the README map).
* Names: tenants appear only as A, B, C; methods only under the paper's names.

**Not reproducible here / not included.**

* Production numbers cannot be recomputed (no data); they are released as aggregates.
* Recall targeting (rho sweep, extra seeds) and extra GUIDE baselines: still running, not included; marked as
  pending placeholders in the README.
* Rounding: the draft prints 106 +- 30 for CE-A100's deployable-gate cost over seeds; the released seeds give
  106.3 +- 29.4.
* The draft attributes the best-local production crossover (about 21) to MaxSim (21.4); RiskFit rho=0.8's point
  crossover is 21.5, so the point envelope is attained by RiskFit rho=0.8. Both round to 21.

**Scans of the final tree.** Same pattern families as above, re-run over every file including the new
aggregates, plus plain-substring checks for: the names and internal ids of the three production tenants (taken
from an internal file and never written down), the company cloud-project name, the storage URL scheme, the
container-registry host, the cloud-project path prefix, the cloud project number, user-home and temporary-directory
prefixes, the local username, the internal code-base name and data-warehouse name, API-key prefixes (OpenAI,
Google, GitHub, AWS, Slack), email addresses and IPv4 addresses. All 0, except the author contact in the
README (one address, deliberate) and the GUIDE checksum fields (`sha256:` in configs, public). No file whose
name marks it as private or not for sharing was opened or copied, and no `.env`, key or credential file was read.

**Tests.** `tests/test_release.py --splits ...`: all checks pass (production granularity and table spot checks,
GUIDE snapshot numbers, figure regeneration, and a full re-run of `analyze_guide.py` matching the snapshot).
`tests/smoke_test.py` trains a small model and was not run for this sync (no training was allowed).
