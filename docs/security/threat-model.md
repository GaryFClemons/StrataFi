# Threat model

StrataFi is a lab that one person runs on one machine. This document says what it protects,
what it trusts, which controls it has, and what they leave open. It describes the setup this
repository ships: every service on 127.0.0.1, one person, synthetic documents. The
[architecture](../architecture.md) describes the system itself.

## Assumptions

- **One person, one machine.** Every port the launcher and `compose.yaml` open is bound to
  127.0.0.1. The API's development sign-in gives any caller a token for any synthetic user.
  That convenience is what makes the lab single-user.
- **Synthetic data only.** The lab is for invented or public documents, such as the Halcyon
  demo corpus. It isn't hardened for confidential documents, personal data or several users. The
  controls below limit the harm a hostile or sensitive document can do; they aren't a reason to
  load real personal data.
- **Out of scope:** someone with a shell on the machine, who can read `.env` and the database
  anyway; the operating system, Docker, the browser and Ollama.

## What it protects

| Asset | Where it lives | Why it matters |
| --- | --- | --- |
| Documents | The artifact store, under `ws/<workspace>/quarantine/<sha256>` | Their bytes are untrusted, and their text may be sensitive |
| Text derived from them | Parsed, cleaned and chunk artifacts; vectors and indexes, including the `stratafi_index` schema; eval sets; drafted questions; training sets | The same text as the documents. A parsed artifact still holds what Clean later redacts |
| Trained models | The artifact store, as a manifest and its parts | Fine-tuned weights can reproduce text they were trained on |
| Secrets | `.env` (database and object store passwords), `data/keys/` (the signing key), API tokens and webhook secrets in Postgres | The signing key mints any identity, and the database role reads every workspace's rows once it sets its scope |
| Records | Postgres: runs, eval results, audit events | The lab's evidence: IDs, hashes, counts and scores, no text |
| The worker | A process | It holds the database and object store credentials while it reads untrusted bytes, so it's the most exposed process |

## What it trusts

- **Trusted:** the person at the keyboard; the lab's own processes (the API, the worker, the
  model gateway), Postgres and Ollama on the same machine; the code in this repository and its
  locked dependencies.
- **Not trusted:** document bytes, which every parser and archive reader treats as hostile; URLs
  people import from, and every address those URLs resolve to; webhook receivers; custom check
  code in a test suite; hosted providers, which are off unless allowed.

## Threats, controls and what remains

| # | Threat | Controls | What remains |
| --- | --- | --- | --- |
| T1 | A hostile document exploits a parser to read secrets, reach the network or take over the worker | **Scan before parse.** `scan/content-safety` is required in every pipeline spec: the bytes must match the extension; archives are limited (10,000 entries, 512 MiB expanded, 100 times compressed, no nested archives, member names never become paths); size, page, encryption and corruption checks run. Infected files must be excluded. **The parse sandbox.** Each parse runs in its own isolated Python process with a time limit, CPU, memory, file-size and open-file limits, a scratch directory and a small environment. **Confinement** on Linux: a network namespace, Landlock (read only Python, the system's libraries and the one document; write only scratch) and seccomp (no sockets, ptrace, process memory calls, io_uring, bpf, mounts or namespaces); the API and the worker also make themselves undumpable there. `limits.parse_confinement: required` refuses to parse without seccomp, and `python -m stratafi_components.confine` shows what applies on a machine | Confinement is Linux-only, and on Windows only the time limit applies, so on a Windows or macOS host a parser exploit runs as you. `best_effort` is the default. Parsers that call a service, such as `parse/tika`, run in the worker, and the service is the isolation. A kernel exploit beats confinement |
| T2 | Server-side request forgery through URL import or webhooks | The URL guard: `http` and `https` only, no credentials in the URL, ports 80 and 443 by default; every address a name resolves to must be public (no private, loopback, link-local, shared, multicast or reserved ranges, and IPv4 inside IPv6 is checked as IPv4); the connection goes to the checked address, so DNS rebinding can't slip through; each redirect is checked again; proxies are ignored; responses are capped by type and size. `test_ssrf_retest.py` holds dozens of known bypass spellings | `suites.webhook_allow_private: true` lets webhooks reach private addresses, on purpose. Allowed hosted providers are reached on purpose |
| T3 | A query that forgets its workspace filter returns another workspace's rows | Row-level security, forced on the tables' owner, on every workspace table. The API sees nothing until a request names its workspace, and each job sees its own. Services connect as the `stratafi` role, never as a superuser | Not a boundary against code that sets its own scope: any process with the role can. Lab indexes in `stratafi_index` have no policy, and are reached only through an artifact the policy already checked |
| T4 | Document, chunk or query text reaches logs, metrics, audit events or errors | Metadata-only logging, codes instead of values in errors, the log formatter's redaction, Postgres and uvicorn settings that keep statement text and raw paths out, and canary tests in the packages, the API and the worker | A new code path without a canary test. The model gateway has none yet. `scripts/security_report.py canaries` counts them per package |
| T5 | Malware in a document | `scan/clamav` scans with clamd's `INSTREAM` and refuses a pass when the signatures are more than 72 hours old | Optional: it runs only when the pipeline names it and `STRATAFI_CLAMAV_ADDRESS` points at a clamd (the `clamav` compose profile). The demo pipeline doesn't scan for malware |
| T6 | Personal data in a document | `scan/pii` finds SSNs, card numbers and IBANs by default, and emails, phone numbers and IP addresses on request, each validated (Luhn, mod 97, number ranges); its outcome `pii` excludes the document by default. `clean/redact` masks the spans, as a mask, a typed placeholder or a keyed hash. A parsed document whose Clean step redacted something is served to owners only. Suites' `no_findings` checks assert nothing reached the chunks. Reports hold counts by category, never values | Pattern-based: names, addresses and personal facts in prose aren't found. Neither component is in the default pipeline. Owners can read pre-redaction text |
| T7 | Stolen secrets | `.env` and `data/` are ignored by Git. `scripts/dev_setup.py` generates random secrets and writes `.env` and the key readable only by their owner (on POSIX file systems). Configuration names secrets by variable and never stores them. The database role isn't a superuser. With SeaweedFS, the services' user reaches one bucket, and only the administrator creates it. API tokens are stored hashed. Identity tokens last five minutes at most | On Windows, file modes don't protect `.env` or the key. Anyone who reads them can mint any identity and reach the database. Webhook signing secrets are stored in Postgres unless `STRATAFI_SUITES__WEBHOOK_SECRET_KEY` derives them |
| T8 | Someone else reaches the lab | Every port is bound to loopback. The development sign-in exists only outside `prod`, and `dev_setup.py token` refuses to mint tokens when `STRATAFI_ENV=prod`. The web app renders document text and model output as text, never as HTML, and pdf.js runs without XFA or fetches | Anyone who reaches the API can sign in as any synthetic user. Don't publish its port; use an SSH tunnel to reach the lab from elsewhere |
| T9 | Custom check code escapes its sandbox | Off by default, behind three gates: `suites.sandbox.backend: docker`, the sandbox's review reference in `suites.sandbox.review`, and the workspace's `custom_checks` feature, which only an owner turns on. Each check runs in its own container: no network, a read-only root, no capabilities, no privilege gain, the `nobody` user, process, memory, CPU and output limits, the run's data read-only, and gVisor when Docker has it | Starting containers needs Docker's socket, which is root on the host. A worker in a container (the `app` profile) has no socket, so custom checks stay off there; giving it one needs a socket proxy limited to the sandbox's calls, or rootless Docker |
| T10 | Text leaves the machine | The model gateway and Ollama run locally. Hosted parsers, rerankers and embedders refuse to run until `providers.allowed` names them. Each run records which providers received which documents, and pipeline maps mark every call that sends text out | Allowing a provider sends it document text |
| T11 | Data outlives its deletion | Purge by lineage: an owner's `DELETE /api/v1/documents/{id}`, `stratafi-store purge-document` and `purge-workspace` delete every artifact derived from a document, including training sets and the models trained on them, which are withdrawn. `retention.document_days`, `artifact_days` and `audit_days` purge by age; they're unset by default | Run records keep the IDs and hashes of purged documents, backups keep data until they're deleted, and downloaded exports and models are outside the lab |
| T12 | A vulnerable or unlicensable dependency | `uv.lock` and `web/package-lock.json` pin every version. `scripts/security_report.py audit` checks both against OSV, and `licenses --check` fails on a license nobody reviewed. AGPL components, such as `parse/pymupdf4llm`, refuse to run until `licenses.allowed` names their license | The reports run by hand, not in CI |
| T13 | A change nobody can trace | Audit events for every route that changes something; `test_audit_coverage.py` fails when a write route records none | Reads aren't audited |

## Synthetic data only

Use the Halcyon demo corpus, other invented documents, or public ones. Don't load personal data
or confidential documents. If that ever changes, the work starts here:

- run on Linux with `limits.parse_confinement: required`, in `prod` mode, which also requires
  `observability.metrics_token`
- put a real sign-in in front of the API, and turn off the development sign-in
- add `scan/clamav` and `scan/pii` to every pipeline, with clamd's signatures kept current
- keep secrets in a secret store rather than `.env`, with `STRATAFI_SUITES__WEBHOOK_SECRET_KEY`
  set
- set retention, and decide how long backups live

## Checking the controls

```bash
uv run python -m stratafi_components.confine
```

```bash
uv run python scripts/security_report.py all
```

The first prints what confinement applies on this machine, and exits 1 if a hostile parser
reached anything. The second prints the dependency audit, the license report and the canary
count; its audit sends the locked names and versions, and nothing else, to OSV's API.

The unit tier covers the URL guard, the archive limits, the PII detector and the log
formatter. The integration tier covers row-level security, purges and the canaries that need a
database. The system tier runs the confinement self-test in the app image and escape attempts
against the custom-check sandbox ([local development](../guides/local-development.md#tests)).
