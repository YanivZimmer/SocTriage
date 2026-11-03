# Scrub report

This repository was assembled from scratch for release. No history was carried over from the internal code
base, and no files were copied wholesale except the vendored, Apache-licensed FastFit trainer
(`riskfit/models/fastfit/`, comments cleaned). This report lists what was left out and how the tree was
checked. It records categories and counts only, never the matched values.

## Excluded by construction

| Category | Treatment |
|---|---|
| Production / tenant data: alerts, alert text, labels, per-alert predictions, perturbed samples | not included; only the schema is documented (README, `riskfit/serialization.py`) |
| Trained adapters, checkpoints, caches, logs, notebooks | not included; `.gitignore` keeps `data/guide_raw/`, splits, predictions, `cache/`, `results/` out of git |
| The public GUIDE dataset | not redistributed; `scripts/download_guide.sh` fetches and checksums it |
| Customer / tenant names and identifiers | removed; the code never names a tenant |
| Internal company, product, service and repository names | removed or renamed (modules rewritten under the `riskfit` package) |
| Employee names, usernames, home and scratch paths | removed; every path is a command-line argument relative to the repository |
| Cloud project ids, buckets, data-warehouse tables, container images, job ids | removed, along with the cloud launchers, data exporters, tenant registry and serving code |
| Chat / ticket references, internal commit hashes, experiment-ledger ids | removed from code and comments |
| Secrets (API keys, tokens, service-account files, `.env`) | none present. Clients read keys only from environment variables (`OPENAI_API_KEY`, `TYPESAFE_API_KEY`, `ANTHROPIC_API_KEY`). No key file or `.env` file was opened while the release was built |
| Paper author block | left as a TODO in the README |

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

