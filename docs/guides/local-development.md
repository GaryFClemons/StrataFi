# Local development

How to run StrataFi on your own machine, load the demo corpus, and run the tests. The
[architecture](../architecture.md) describes what the pieces do.

## What runs where

On one machine, by default:

| Process | Address | Started by |
| --- | --- | --- |
| Postgres 16 with pgvector | `127.0.0.1:5442` | Docker Compose, or WSL on Windows |
| Ollama | `127.0.0.1:11434` | The Ollama app |
| Model gateway | `http://127.0.0.1:8091` | `scripts/local.py up` |
| API | `http://127.0.0.1:8000` (contract at `/api/v1/docs`) | `scripts/local.py up` |
| Worker | No port | `scripts/local.py up` |
| Web app | `http://127.0.0.1:5190/stratafi/` | `scripts/local.py up` (Vite) |

Artifacts are files under `data/artifacts`. ClamAV, Tika and SeaweedFS are optional, each in a
Compose profile of its own. Every port is bound to 127.0.0.1.

## What you need

- **uv 0.13.5**, the version that writes `uv.lock`'s format, and **Python 3.12**
  (`uv python install 3.12` fetches it).
- **Node 24**, as `.nvmrc` says, for `web/`.
- **[Ollama](https://ollama.com)** with the two models the lab uses by default:

  ```bash
  ollama pull qwen2.5:3b && ollama pull nomic-embed-text
  ```

- **Docker with Compose v2** for Postgres. On Windows without Docker, a WSL distribution
  instead ([below](#in-wsl-without-docker)).
- **Optional:** the gateway's `rerank` extra, for cross-encoder reranking
  ([the model gateway](#the-model-gateway)), and Tesseract, for the OCR parser and its tests.

## First run

Install the Python workspace and the web app's packages:

```bash
uv sync --all-packages --locked
```

```bash
npm ci --prefix web
```

Write `.env` and the development signing key:

```bash
uv run python scripts/dev_setup.py
```

Start Postgres, with Docker:

```bash
docker compose up -d --wait
```

Check what the lab needs, then start it. `up` creates and migrates the database, then runs the
gateway, the API, the worker and the web app until you press Ctrl+C:

```bash
uv run python scripts/local.py doctor
```

```bash
uv run python scripts/local.py up
```

In a second terminal, load the demo corpus, index it and evaluate it:

```bash
uv run python scripts/local.py seed
```

Then open `http://127.0.0.1:5190/stratafi/`. The web app signs you in as the local owner by
itself.

## Postgres

### With Docker Compose

`docker compose up -d --wait` starts only Postgres: the `pgvector/pgvector:pg16` image, pinned
by digest, on port `STRATAFI_PG_PORT` (5442). Its data lives in the `postgres-data` volume.
Postgres runs with `log_min_error_statement=panic` and `log_error_verbosity=terse`, so a failed
statement's text and the key values in `DETAIL` lines never reach its log.

Postgres keeps the admin password it was created with. If you write a new `.env` with new
secrets, run `docker compose down -v` first. That deletes the development data.

### In WSL, without Docker

On a Windows machine where Docker can't run, `scripts/wsl/postgres.sh` installs PostgreSQL 16
and pgvector in a WSL distribution (Ubuntu 22.04 or 24.04) from the PostgreSQL project's apt
repository. It sets the port from `.env`, keeps statement text out of the logs as Compose does,
and creates the admin role `.env` names. From the repository, in a Windows terminal, after
`dev_setup.py`:

```bash
wsl -d Ubuntu-24.04 -u root -- bash scripts/wsl/postgres.sh
```

Windows reaches it on 127.0.0.1 through WSL's localhost forwarding. Then name the distribution
in `.env`, so that `scripts/local.py up` starts its Postgres and keeps WSL awake while the lab
runs:

```text
STRATAFI_WSL_DISTRO=Ubuntu-24.04
```

The script is safe to run again, for example after `.env` gets a new admin password.

### The database and its role

`scripts/local.py up` runs both of these first. To run them yourself:

```bash
uv run --env-file .env stratafi-store bootstrap
```

```bash
uv run --env-file .env stratafi-store migrate
```

`bootstrap` connects with the admin account (`STRATAFI_PG_ADMIN_DSN`). It creates the
`stratafi` role, which can log in and do nothing else, a database it owns, and the `vector`
extension. Services connect as that role (`STRATAFI_DATABASE_URL`), never as a superuser,
because superusers skip row-level security. Both commands are safe to run again; `bootstrap`
also resets the role's password to the one in `.env`.

## Settings and .env

`dev_setup.py` builds `.env` from `.env.example`. Each `__GENERATE_SECRET__` value gets a random
secret the first time, `${NAME}` refers to another variable, and values already in `.env` are
kept, so it's safe to run again. It also writes the development key pair to `data/keys/`. Git
ignores `.env` and `data/`.

| Variables | What for |
| --- | --- |
| `STRATAFI_ENV`, `STRATAFI_CONFIG_OVERLAY` | `dev`, and the overlay that lets the API import from `fixtures/` |
| `STRATAFI_IDENTITY_*`, `STRATAFI_DEV_SIGN_IN_KEY` | The key pair that signs and checks identity tokens, and the development sign-in |
| `STRATAFI_PG_*`, `STRATAFI_DATABASE_URL` | Postgres: the admin account for `bootstrap`, and the `stratafi` role |
| `STRATAFI_MODEL_GATEWAY_URL` | The model gateway; empty means none |
| `STRATAFI_STORAGE__DRIVER`, `STRATAFI_S3_*` | `local` (files under `data/artifacts`) or `s3` |
| `STRATAFI_CLAMAV_ADDRESS`, `STRATAFI_TIKA_URL` | The optional clamd and Tika; empty means none |
| `STRATAFI_TEST_MODEL_URL` | The model tier's endpoint |

The API and the worker read `config/stratafi.yaml`, then the overlay in
`STRATAFI_CONFIG_OVERLAY` on top: mappings merge, and anything else is replaced. `${VAR}`
expands from the environment, and `${VAR:-default}` gives a default; a missing variable without
one stops the service with an error that names it. `STRATAFI_<SECTION>__<KEY>` variables
override both files, for example `STRATAFI_WORKER__LEASE_SECONDS=30` or
`STRATAFI_WORKER__SLOTS='{"cpu": 4, "gpu": 1, "io": 4}'`.

## The launcher

`scripts/local.py` runs the lab on one machine.

- **`doctor`** checks `.env`, that Postgres answers, that Ollama answers and has `qwen2.5:3b`
  and `nomic-embed-text`, and that `web/node_modules` and `npm` are there. Each problem comes
  with its fix, and it exits 1 if there is one.
- **`up`** starts Postgres in WSL when `.env` names `STRATAFI_WSL_DISTRO`, runs `doctor`, then
  `bootstrap` and `migrate`. It starts the gateway, the API, the worker and the web app as child
  processes, with each line of their output prefixed by its name, and prints the lab's address
  once the API answers. If one of them stops, it stops the rest. Ctrl+C stops everything.
- **`seed`** loads the demo corpus. See [Seeding the demo](#seeding-the-demo).

Each service also runs on its own, which helps when you restart one:

```bash
uv run --env-file .env stratafi-api
```

```bash
uv run --env-file .env stratafi-worker
```

```bash
uv run --env-file .env stratafi-gateway
```

```bash
cd web && npm run dev
```

The worker claims jobs in `worker.slots` threads per resource class (`cpu: 2, gpu: 1, io: 4`).
Ctrl+C or SIGTERM stops it gracefully: it claims nothing new, gives running jobs 30 seconds, and
then hands them back to the queue. Vite proxies `/api` to `http://127.0.0.1:8000`, or to
`STRATAFI_API_URL`. `STRATAFI_BASE_PATH=/a/b/ npm run build` builds the web app for another base
path.

## The model gateway

`services/gateway` serves the model gateway contract on `http://127.0.0.1:8091`: embeddings and
chat from Ollama, and reranking from Hugging Face cross-encoders. List what it serves:

```bash
curl http://127.0.0.1:8091/v1/models
```

Every pulled Ollama model is served under its own name, so a spec can name `nomic-embed-text`
or `qwen2.5:3b` directly. There are three aliases too: `embed-default` (`nomic-embed-text`),
`chat-default` (`qwen2.5:3b`) and, with the `rerank` extra, `rerank-default`
(`cross-encoder/ms-marco-MiniLM-L-6-v2`). An Ollama model's revision is its weights' digest, so
pulling a model again changes the revision and the cache keys that depend on it.

Ollama serves no rerankers. To rerank, install the gateway's `rerank` extra, which brings
sentence-transformers and torch. `uv sync` removes what it wasn't asked for, so pass the same
extras every time you sync:

```bash
uv sync --all-packages --locked --extra rerank
```

The gateway reads `STRATAFI_GATEWAY_*` variables, documented in
`services/gateway/src/stratafi_gateway/settings.py`. Put them in `.env` for the launcher. For
example, to add an alias and a second reranker:

```text
STRATAFI_GATEWAY_ALIASES=embed-local=nomic-embed-text
STRATAFI_GATEWAY_RERANKERS=rerank-default=cross-encoder/ms-marco-MiniLM-L-6-v2,bge-reranker-base=BAAI/bge-reranker-base
```

Without Ollama, the fake gateway serves the same contract with hashed vectors and word-overlap
reranking, on the same port:

```bash
uv run python -m stratafi_testing.fake_gateway
```

## Seeding the demo

`scripts/local.py seed` signs in as the local owner and loads the Halcyon Coffee demo corpus
(`fixtures/demo-corpus`): 24 invented documents and 88 questions.

1. It creates the corpus `halcyon`, uploads the documents with their manifest, and freezes it.
2. It saves the pipeline `local-baseline`: content checks and deduplication, the default parse
   route, `chunk/sections` at 200 to 400 tokens, `nomic-embed-text` through the gateway, and an
   exact and a BM25 index. Then it runs it. The first embedding takes a while.
3. It imports the eval set `halcyon-retrieval`, with 30% of its items held out as the test
   split.
4. It evaluates three retrieval configs, `local-keyword`, `local-dense` and `local-hybrid`, and
   a fourth, `local-hybrid-rerank`, when the gateway serves `rerank-default`.

It ends with a Compare link and a Chat link. Running it again reuses everything already
cached.

`scripts/demo_seed.py` seeds a lab with no models at all: the same corpus, embedded with
`embed/fake-hash` and reranked with `rerank/fake-overlap`, so the screens have something to show.
Its numbers say nothing about a model. It seeds the `app` profile's API by default;
`--url http://127.0.0.1:8000` seeds the launcher's.

## The command line and the SDK

The `stratafi` command and the Python SDK (`packages/sdk`) need the API's address. With neither
`STRATAFI_TOKEN` nor `STRATAFI_DEV_SIGN_IN` set, they sign in as the local owner, as the web app
does:

```bash
export STRATAFI_URL=http://127.0.0.1:8000
```

In PowerShell, `$env:STRATAFI_URL = "http://127.0.0.1:8000"`. With the seed's names:

```bash
uv run stratafi run start --pipeline local-baseline --corpus halcyon@latest --wait
```

```bash
uv run stratafi eval run --run RUN_ID --retrieval local-hybrid --eval-set halcyon-retrieval@latest --split dev --wait
```

```bash
uv run stratafi compare EVAL_RUN_A EVAL_RUN_B --format markdown
```

The examples default to the seed's names (the corpus `halcyon`, the pipeline `local-baseline`,
the retrieval config `local-hybrid` and the eval set `halcyon-retrieval`), so they run as soon as
`seed` has. `examples/chunk_size_sweep.py` sweeps the chunk size and prints each size's recall@5
and MRR@10:

```bash
uv run python examples/chunk_size_sweep.py --sizes 256,384,512,800
```

`examples/suites/policies-regression.yaml` is a test suite to start from: thresholds, a golden
query, and a check against the last passing run:

```bash
uv run stratafi test run examples/suites/policies-regression.yaml --junit results.xml
```

The contract fixtures load with the command line too. `baseline-v1` scans with ClamAV and the
PII scanner and embeds with the alias `embed-local`, and `hybrid-v1` reranks with
`rerank-default`. To run them as they are, start the `clamav` profile and set
`STRATAFI_CLAMAV_ADDRESS`, add the alias `embed-local=nomic-embed-text` to the gateway, and
install the `rerank` extra:

```bash
uv run stratafi pipeline push contracts/fixtures/pipeline.v1/baseline-v1.yaml
```

```bash
uv run stratafi retrieval push contracts/fixtures/retrieval.v1/hybrid-v1.yaml
```

Without the seed, the corpus and the eval set load the same way:

```bash
uv run stratafi corpus create halcyon
```

```bash
uv run stratafi corpus upload halcyon --manifest fixtures/demo-corpus/corpus/manifest.yaml
```

```bash
uv run stratafi corpus freeze halcyon
```

```bash
uv run stratafi evalset import fixtures/demo-corpus/retrieval.yaml --name halcyon-retrieval --split-test 0.3
```

The [SDK's README](../../packages/sdk/README.md) lists every command and the client's resources.
Exit codes: `0` passed, `1` a check failed, `2` a usage or configuration error, `3` the lab
failed or couldn't be reached.

## Optional services and the app profile

Each optional service has a Compose profile of its own. Start it, then set its variable in
`.env` and restart the lab:

| Profile | Service | Port | Set in `.env` |
| --- | --- | --- | --- |
| `clamav` | ClamAV's clamd, for `scan/clamav` | 3320 | `STRATAFI_CLAMAV_ADDRESS=127.0.0.1:3320` |
| `tika` | Apache Tika, for `parse/tika` | 9998 | `STRATAFI_TIKA_URL=http://127.0.0.1:9998` |
| `seaweedfs` | SeaweedFS, an S3 API for the artifacts | 9100 | `STRATAFI_STORAGE__DRIVER=s3` |

```bash
docker compose --profile clamav up -d --wait
```

- **ClamAV** loads its signatures for a minute or two. freshclam keeps them current in the
  `clamav-signatures` volume, and `scan/clamav` refuses signatures more than 72 hours old, so
  the container needs to reach ClamAV's mirror.
- **SeaweedFS** runs as one `weed server`. Its users come from a file Compose writes from `.env`:
  the services' user may read, write, list and tag in the bucket and nothing else, and an
  administrator, used only by `seaweedfs-init`, creates the bucket.

### Everything in containers

The `app` profile runs the API, the worker, the model gateway and the web app in containers,
built from this checkout. The API bootstraps and migrates the database before it starts.

```bash
docker compose --profile app build
```

```bash
docker compose --profile app up -d --wait
```

The web app is then at `http://127.0.0.1:4100/stratafi/`, the API at port 8100 and the gateway
at 8191. The gateway reaches Ollama on the host through `host.docker.internal`. On Linux, Ollama
must listen beyond loopback for that (`OLLAMA_HOST=0.0.0.0`). The image leaves out the `rerank`
extra, so reranking needs the gateway on the host. Artifacts go to the `artifacts` volume, or
to SeaweedFS with `STRATAFI_STORAGE__DRIVER=s3`.

## Tests

| Tier | Command | Needs |
| --- | --- | --- |
| Unit | `uv run pytest -p no:cacheprovider -m "not integration and not model and not system"` | Nothing |
| Integration | `uv run pytest -p no:cacheprovider -m integration` | Docker, or local services |
| Model | `uv run pytest -p no:cacheprovider -m model` | `STRATAFI_TEST_MODEL_URL` |
| System | `STRATAFI_TEST_SYSTEM=1 uv run pytest -p no:cacheprovider -m system` | Docker |
| Browser | `STRATAFI_TEST_BROWSER=1 uv run pytest -p no:cacheprovider -m browser` | A built `web/dist`, Playwright and Chromium, and the integration tier's Postgres |
| Fine-tuning | `STRATAFI_TEST_HF_MODELS=1 uv run pytest -p no:cacheprovider -k fine_tune` | The `sentence-transformers` extra; downloads two small models from Hugging Face |

The pytest plugin in `packages/testing` skips a tier whose environment is missing, and says
why. The web app has its own checks:

```bash
cd web && npm run lint && npm run typecheck && npm test && npm run build && npm run budgets
```

### Services for the integration tier

The integration tier never uses your development data. By default, the `compose_services`
fixture starts `compose.yaml` as a separate project, `stratafi-test`, with test-only
credentials and ports Docker picks, and removes it, volumes included, when the session ends.
`STRATAFI_KEEP_SERVICES=1` keeps it for the next session. Tests never pull images, so pull them
once:

```bash
docker compose --profile seaweedfs --profile clamav --profile tika pull
```

Instead, or without Docker, point the tier at local services, one set of variables per
service. A service whose variables are set never starts in Docker, and a service with neither
makes its tests skip.

| Service | Variables |
| --- | --- |
| Postgres 16 with pgvector | `STRATAFI_TEST_PG_ADMIN_DSN`: a superuser's URL |
| ClamAV | `STRATAFI_TEST_CLAMD`: `host:port`, or `fake` for the in-process fake that flags only the EICAR test file |
| SeaweedFS or another S3 API | `STRATAFI_TEST_S3_ENDPOINT`, `STRATAFI_TEST_S3_ACCESS_KEY`, `STRATAFI_TEST_S3_SECRET_KEY`, `STRATAFI_TEST_S3_BUCKET` (the bucket must exist) |
| Tika 3.2 | `STRATAFI_TEST_TIKA_URL` |

With Postgres in WSL, the admin URL in `.env` works. Each test creates its own database,
`t_<random>`, owned by a role that isn't a superuser, so tests run under row-level security as
the services do, and drops it afterwards. The `stratafi` database is never touched.

```bash
export STRATAFI_TEST_PG_ADMIN_DSN="$(grep '^STRATAFI_PG_ADMIN_DSN=' .env | cut -d= -f2-)" STRATAFI_TEST_CLAMD=fake
```

CI does the same with a Postgres service container and the fake clamd.

### Other tiers

- **Model:** point it at the gateway, and name an embedding model it serves. Its rerank test
  needs the `rerank` extra.

  ```bash
  STRATAFI_TEST_MODEL_URL=http://127.0.0.1:8091 STRATAFI_TEST_EMBED_ALIAS=nomic-embed-text uv run pytest -p no:cacheprovider -m model
  ```

- **System:** escape attempts against the custom-check sandbox, whose image the tests build as
  `stratafi/check-sandbox:test`, and the parser confinement self-test in the app image, which
  you build first with `docker compose --profile app build api`.
- **Browser:** each flow starts its own API, worker and fakes on a fresh database, serves
  `web/dist` at `/stratafi/`, and drives a script from `web/e2e` in Chromium. The scripts load
  the `playwright` library, which `web/` doesn't install, from wherever Node finds it.
  `STRATAFI_CHROMIUM` names a Chromium binary, and `STRATAFI_SCREENSHOTS=<dir>` saves a
  screenshot after each step.

## Contracts and fixtures

CI fails when a generated file is stale. After changing the API, export the OpenAPI document,
then regenerate the SDK's models and the web client:

```bash
uv run python scripts/export_openapi.py && uv run python scripts/export_sdk_models.py && (cd web && npm run api:generate)
```

After changing a spec or artifact model, export the JSON Schemas in `contracts/schemas`:

```bash
uv run python scripts/export_contracts.py
```

Each export script takes `--check`, as CI runs it, and `npm run api:check` checks the web
client. After changing a parser, rewrite its goldens. They cover the parsers this machine can
run, so missing extras leave theirs out:

```bash
uv run python scripts/parser_goldens.py
```

After changing the format fixtures' script or the demo documents it renders, write the fixtures
again, then the goldens:

```bash
uv run python scripts/format_fixtures.py && uv run python scripts/parser_goldens.py
```

The multi-format variants come from `scripts/corpus_variants.py`
([the fixtures' README](../../fixtures/README.md)).

## The database

One baseline migration, `0001_baseline`, creates the schema. A schema change goes in a new,
hand-written migration:

```bash
uv run stratafi-store revision -m "add runs"
```

Write its `upgrade()`, and update `stratafi_store/tables.py` to match:
`test_the_migrations_build_exactly_the_metadata` fails until the two agree.
`stratafi-store current` shows the applied and the head revision.

Deletion and backups work from the command line too. An owner's
`DELETE /api/v1/documents/{id}` purges a document by lineage, as `purge-document` does:

```bash
uv run --env-file .env stratafi-store purge-document --workspace <id> --document <id>
```

```bash
uv run --env-file .env stratafi-store purge-workspace --workspace <id> --confirm <slug>
```

`retention.document_days`, `artifact_days` and `audit_days` make the worker purge what has
expired, and `stratafi-store retention` runs one sweep now. A backup is a directory with the
database's dump, every object and their checksums. It needs `pg_dump` and `pg_restore` 16 on
the path:

```bash
uv run --env-file .env stratafi-store backup --out data/backups/latest
```

```bash
uv run --env-file .env stratafi-store verify --from data/backups/latest
```

Restore into an empty database (`bootstrap`, without `migrate`) and an empty store. It migrates
afterwards:

```bash
uv run --env-file .env stratafi-store restore --from data/backups/latest
```

## Custom checks

A test suite's custom checks (`python: checks/tables.py:function`) run only in a sandbox: one
container per check, with no network and no credentials. Three gates must all be open, and
until they are, each custom check ends as an error, `custom_checks_disabled`. Build the image:

```bash
docker build -f docker/sandbox/Dockerfile -t stratafi/check-sandbox:1 .
```

Then point the worker at Docker and record the sandbox's security review in `.env`
(`STRATAFI_SUITES__SANDBOX__BACKEND=docker`, `STRATAFI_SUITES__SANDBOX__REVIEW=<reference>`), and
have an owner turn custom checks on for the workspace: Tests, then Automation, or
`PUT /api/v1/workspace/features/custom_checks` with `{"enabled": true}`. With synthetic
documents on your own machine, any reference lets you try the flow.

The worker runs `docker run` for each check, so run it on the host, with the Docker CLI. The
`app` profile's worker has no Docker socket. `stratafi test run suite.yaml` sends the files the
checks name from beside the suite file, and `stratafi.checks.CheckContext.from_records` tries a
check locally first.

## Post-training

Training is lab-only: it's refused while `models.in_process` is off.

1. Run a dev eval run, then mine it into a training set: the Training screen, or
   `uv run stratafi train mine EVAL_RUN_ID --wait`.
2. Train. `train/query-adapter` (numpy) runs on the `cpu` slots. `train/sentence-transformers`
   and `train/cross-encoder` need the `sentence-transformers` extra and run on the `gpu` slot,
   on a CPU-only machine too. They download their base model from Hugging Face the first time.
3. Use the model. An adapter goes in a retrieval config, as
   `retrievers.dense: {top_n: 50, adapter: MODEL_ID}`. An embedder or a reranker is
   `model: trained/MODEL_ID` of `embed/sentence-transformers` or `rerank/cross-encoder`.

`uv run stratafi model download MODEL_ID --output model.zip` writes a model's files with its
card.

## Metrics and the load test

Prometheus can read the API at `/api/v1/metrics` (requests by route and the queue), and the
worker at `observability.worker_metrics_port` when it's set (jobs, stages and providers).
`scripts/load_test.py` runs a synthetic corpus through the API and a worker in one process, then
an evaluation, against a new database:

```bash
STRATAFI_TEST_PG_ADMIN_DSN=postgresql://... uv run python scripts/load_test.py --documents 1000 --chunks 100 --queries 1000
```

## Development tokens

The API accepts identity tokens signed with the key in `data/keys/`. Mint one for a synthetic
user (`owner`, `editor`, `viewer` or `guest`) and send it as the `X-Stratafi-Identity` header:

```bash
curl -H "X-Stratafi-Identity: $(uv run python scripts/dev_setup.py token --tenant local --user owner)" http://127.0.0.1:8000/api/v1/me
```

A token lasts five minutes at most, and its tenant names the workspace, which the tenant's first
token creates. `--sub`, `--roles` and `--groups` mint any other identity. `dev_setup.py` refuses
to mint tokens when `STRATAFI_ENV=prod`, and the API's development sign-in, which the web app and
the SDK use, exists only outside `prod`.
