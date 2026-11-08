# stratafi-api

StrataFi's REST API. Every route lives under `/api/v1`, and the interactive contract is at
`/api/v1/docs`. `contracts/openapi.json` is its exported OpenAPI document, from which the web
client and the SDK's models are generated.

```bash
uv run --env-file .env stratafi-api
```

It listens on `127.0.0.1:8000` by default (`api.host`, `api.port`).

- **Errors** are RFC 9457 problems (`application/problem+json`) that never echo input, and every
  response carries an `X-Request-Id`.
- **Identity:** a short-lived token in `X-Stratafi-Identity`, or a scoped API token as a
  bearer token. Outside `prod`, `POST /api/v1/dev/token` signs the web app and the SDK in as a
  synthetic user; it isn't in the contract.
- **Live progress:** `GET /api/v1/events` streams the workspace's job and run events as
  server-sent events, from Postgres `LISTEN`/`NOTIFY`.
- **Chat:** `POST /api/v1/chat` streams a grounded, cited answer as server-sent events.
- **Health and metrics:** `GET /api/v1/health/live`, `GET /api/v1/health` (the database and the
  artifact store) and `GET /api/v1/metrics` for Prometheus.

Requests are logged as one metadata-only line each; uvicorn's access log is off, since it would
record raw paths and query strings. [The architecture](../../docs/architecture.md) describes the
rest.
