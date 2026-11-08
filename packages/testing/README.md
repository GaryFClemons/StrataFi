# stratafi-testing: fakes and fixtures

Test tools only, never deployed. The fixtures load automatically wherever this package is
installed, through the `pytest11` entry point in `pyproject.toml`.

## Tiers

Every test belongs to one tier, and the plugin skips a tier whose environment is missing,
saying why:

| Marker | Tier | Runs when |
| --- | --- | --- |
| none | Unit | Always |
| `integration` | Integration | Docker answers, or local services are set (`compose_services`) |
| `model` | Model | `STRATAFI_TEST_MODEL_URL` names an OpenAI-compatible endpoint, such as the local gateway |
| `system` | System | `STRATAFI_TEST_SYSTEM=1`, with Docker |
| `browser` | Browser (also `integration`) | `STRATAFI_TEST_BROWSER=1`, with `web/dist` built and Playwright's Chromium installed |

The [local development guide](../../docs/guides/local-development.md#tests) has the commands.

## Fixtures

| Fixture | Gives you |
| --- | --- |
| `repo_root` | The repo root |
| `canary` | A `CanaryCheck` with a fresh canary: `.value`, `.assert_absent(text)`, `.assert_not_logged()`. It uses `capfd`, so a test can't also use `capsys` |
| `dev_identity` | A throwaway key pair, and tokens for the synthetic users |
| `compose_services` | `services = compose_services("postgres", "seaweedfs")`, then `services.pg_admin_dsn(db)`, `services.s3_endpoint()`, `services.s3_bucket()`, `services.clamd_address()` or `services.tika_url()`. Integration tests only |
| `fresh_database` | `url = fresh_database()`: an empty database with pgvector, owned by a role that isn't a superuser, dropped after the test |
| `migrated_engine` | An engine on a fresh database with every migration applied |
| `serve_app` | `url = serve_app(app)` runs an ASGI app on 127.0.0.1 on a free port |
| `model_endpoint` | `STRATAFI_TEST_MODEL_URL`, for the model tier |

`compose_services` starts this repo's `compose.yaml` as its own project, `stratafi-test`, with
test-only credentials and ports Docker picks, and removes it at the end of the session unless
`STRATAFI_KEEP_SERVICES=1`. It never pulls or builds images. Without Docker, set each service's
local variables
([the guide](../../docs/guides/local-development.md#services-for-the-integration-tier)).

## Canary helpers

`stratafi_testing.canary` enforces "no text in logs": plant `canary.value` where document, chunk
or query text would go, run the code, then call `canary.assert_not_logged()`. It fails if the
canary is in a log message, a field passed with `extra=`, the formatted record, or anything
written to stdout or stderr. `canary.assert_absent(body, ...)` checks response bodies, error
messages and events.

## Other helpers

- `stratafi_testing.lab`: a running lab in the test's process (the API, a worker, fakes for
  clamd and the model gateway), and `prepare_halcyon`, which runs the demo corpus through it.
- `stratafi_testing.fake_gateway`: the model gateway contract with hashed vectors and
  word-overlap reranking (`uv run python -m stratafi_testing.fake_gateway`, on 127.0.0.1:8091).
- `stratafi_testing.clamd`: `ping`, `instream` and `fake_clamd()`, an in-process clamd that
  flags only the EICAR test file (`eicar()`, assembled at runtime).
- `stratafi_testing.users`: synthetic users per tenant (`local`, `tenant0`, `halcyon`): `owner`,
  `editor`, `viewer` and `guest`, whose token subject is `u-<tenant>-<role>`.
