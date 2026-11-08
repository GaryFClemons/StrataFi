# Contributor instructions

StrataFi is a self-hosted lab for retrieval-augmented generation (RAG). People run each ETL step
in a UI (extract, scan and clean, parse, chunk, embed), evaluate retrieval against labelled
questions with confidence intervals, sweep configurations as experiments, track and compare
every run, fine-tune retrieval models on the lab's evidence, and chat with an indexed corpus.
It runs on one machine, for one person, with local models through Ollama.

These instructions are for people and coding agents changing it. Read
[the architecture](docs/architecture.md) before a larger change, and
[the local development guide](docs/guides/local-development.md) to run it.

## Layout

```text
packages/core         stratafi_core: specs, hashing, the component interface, metrics,
                      statistics, settings, identity tokens, metadata-only logging
packages/components   stratafi_components: the built-in components, the parse sandbox, the URL guard
packages/store        stratafi_store: tables, the migration, the job queue, the artifact store,
                      row-level security, purge and backup; the stratafi-store command
packages/sdk          stratafi: the Python client and the stratafi command
packages/testing      stratafi_testing: the pytest plugin, fixtures, fakes and the canary check
services/api          stratafi_api: the REST API under /api/v1, events, chat
services/worker       stratafi_worker: the job handlers
services/gateway      stratafi_gateway: the local model gateway over Ollama and cross-encoders
web/                  the React app: src/lab, src/standalone, unit tests, browser flows in e2e/
contracts/            the OpenAPI document, the JSON Schemas and their fixtures
config/               settings (stratafi.yaml) and overlays
fixtures/             the Halcyon demo corpus, format fixtures and parser goldens
scripts/              setup, the launcher, seeds, exports and reports
docker/               the app, web and custom-check sandbox images
examples/             an SDK script, test suites and a CI workflow
docs/                 the architecture, guides and the threat model
```

## Work safely

- Run `git status --short` first, and keep unrelated work intact.
- Never commit `.env`, anything under `data/`, keys, tokens or other secrets.
- Keep a change to one concern, and update the docs it makes wrong.
- When you finish, report the changed behaviour, the exact checks you ran with their results,
  the checks you skipped and why, and what remains.

## Commands

CI (`.github/workflows/ci.yml`) runs the Python, web and integration checks below on every pull
request. Use uv 0.13.5, so `uv.lock` stays in its format.

Python: sync the locked workspace, lint, and run the unit tier.

```bash
uv sync --all-packages --locked
```

```bash
uv run ruff check && uv run ruff format --check
```

```bash
uv run pytest -p no:cacheprovider -m "not integration and not model and not system"
```

`uv sync` removes packages it wasn't asked for: add the same `--extra` flags each time, such as
`--extra rerank` for the gateway's cross-encoders or `--extra sentence-transformers` for
in-process models and fine-tuning.

Web, from `web/` (Node 24, `.nvmrc`):

```bash
cd web && npm ci && npm run lint && npm run typecheck && npm test && npm run build
```

Contracts: after changing the API, export the OpenAPI document and regenerate the SDK's models
and the web client. After changing a spec or artifact model, export the JSON Schemas in
`contracts/schemas`. CI fails when any of them is stale (each script's `--check`, and
`npm run api:check`).

```bash
uv run python scripts/export_openapi.py && uv run python scripts/export_sdk_models.py && (cd web && npm run api:generate)
```

```bash
uv run python scripts/export_contracts.py
```

Test fixtures: after changing a parser, rewrite its goldens. After changing the format fixtures'
script or the documents it renders, write the fixtures again, then the goldens
([fixtures/README.md](fixtures/README.md)).

```bash
uv run python scripts/parser_goldens.py
```

```bash
uv run python scripts/format_fixtures.py && uv run python scripts/parser_goldens.py
```

Running the lab: `dev_setup.py` writes `.env` and the development signing key, Compose starts
Postgres (on Windows without Docker, `scripts/wsl/postgres.sh` runs it in WSL), and the
launcher creates and migrates the database, then runs the gateway, the API, the worker and the
web app until Ctrl+C. `seed` loads the demo corpus, in a second terminal.

```bash
uv run python scripts/dev_setup.py
```

```bash
docker compose up -d --wait
```

```bash
uv run python scripts/local.py up
```

```bash
uv run python scripts/local.py seed
```

The integration tier starts its own Compose project (`stratafi-test`), so it never touches the
development data. Without Docker, it runs against local services named by `STRATAFI_TEST_*`
variables ([the guide](docs/guides/local-development.md#services-for-the-integration-tier)).

```bash
uv run pytest -p no:cacheprovider -m integration
```

The model tier needs `STRATAFI_TEST_MODEL_URL`, such as the gateway's address. Fine-tuning
tests run only with the `sentence-transformers` extra and `STRATAFI_TEST_HF_MODELS=1`, since
they download models from Hugging Face. The system tier needs Docker: it builds the custom-check
sandbox's image for its escape tests
(`STRATAFI_TEST_SYSTEM=1 uv run pytest -p no:cacheprovider -m system`). To build the sandbox
image yourself:

```bash
docker build -f docker/sandbox/Dockerfile -t stratafi/check-sandbox:1 .
```

## Rules

- **No text in logs.** Document, chunk and query text never goes into logs, metrics, audit
  events, job progress, events or error messages. Log fixed format strings with IDs, hashes,
  codes and counts, and raise errors with codes and field paths, never values. The packages,
  the API and the worker have canary tests (`stratafi_testing.canary`): a new code path that
  touches text gets one.
- **Synthetic data only.** Use the Halcyon demo corpus, other invented documents or public
  ones. Never commit personal data, customer documents or anything confidential.
- **Model calls go through the model gateway by default.** In-process models are lab-only and
  stay behind `models.in_process`. Hosted providers stay off until `providers.allowed` names
  them, and a component that sends text out declares it in its `egress`.
- **Contracts stay current.** `contracts/` holds the OpenAPI document and a versioned JSON
  Schema for every spec and artifact. Both are generated from the code: regenerate them with
  the commands above. The fixtures beside them are written by hand, and a unit test validates
  each one. A breaking change to a schema makes a new version (`.v2`) rather than changing
  `.v1`.
- **Keep `uv.lock` in uv 0.13.5's format.** Lock with that version, and keep every dependency
  installable on Linux x86_64 and Windows AMD64 (`required-environments`). Heavy, Linux-only
  extras carry a `sys_platform == 'linux'` marker.
- **Components are versioned.** Bump a component's `version` whenever its output could change,
  since the version is part of every cache key; for a parser, rewrite its goldens too. A new
  component declares its license, platforms, egress and the services it uses.
- **Schema changes go in a new migration** after `0001_baseline`, with `stratafi_store/tables.py`
  updated to match.
- **One person, one machine.** The UI and the SDK sign in as the local owner (`local:owner`)
  through the development sign-in; tokens travel as `X-Stratafi-Identity`. Keep every port on
  127.0.0.1.
