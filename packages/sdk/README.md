# stratafi (SDK and CLI)

The Python client and the `stratafi` command line for a StrataFi lab. The models in
`stratafi/models.py` are generated from `contracts/openapi.json`
(`scripts/export_sdk_models.py`), so the SDK and the web client share one contract; a thin layer
adds references by name, waiting, variants and results as data.

```python
from stratafi import Client

client = Client.from_env()
base = client.pipelines.get("local-baseline")

rows = []
for max_tokens in (256, 384, 512, 800):
    spec = base.with_params("chunk", max_tokens=max_tokens, min_tokens=max_tokens // 2)
    run = client.runs.start(spec, corpus="halcyon@latest", wait=True)
    result = client.evals.run(
        run, retrieval="local-hybrid", eval_set="halcyon-retrieval@1", split="dev", wait=True
    )
    rows.append({"max_tokens": max_tokens, **result.metrics.select("recall@5", "mrr@10")})

print(client.reports.table(rows))
```

The names are the ones the demo seed (`scripts/local.py seed`) saves.
`examples/chunk_size_sweep.py` is this script with options.

## Configuration

- `STRATAFI_URL`: the API's address (`/api/v1` is added).
- `STRATAFI_TOKEN`: an API token (`sft_...`), sent as `Authorization: Bearer`; or an identity
  token, sent as `X-Stratafi-Identity`. An owner makes API tokens on the Tests screen or with
  `POST /api-tokens`, with scopes (`read`, `runs:write`, `tests:run`) and an expiry of at most a
  year; the token is shown once, and only its hash is stored. CI needs `read` and `tests:run`.
- `STRATAFI_DEV_SIGN_IN`: instead of a token, `tenant:role` (`tenant0:editor`); with neither set,
  `local:owner`, the lab's local owner. The client asks the API's development sign-in route for
  short-lived tokens and renews them as they age.

## The client

| Resource | Does |
| --- | --- |
| `client.corpora` | `create`, `upload` (files, folders, a manifest), `freeze`, `get`, `list`, `version("halcyon@latest")` |
| `client.pipelines`, `client.retrieval` | `get("name@N")`, `validate`, `push`, `list`; `Pipeline.with_params(stage or component ID, **params)` makes a variant |
| `client.runs` | `start(pipeline, corpus=..., until=..., sample=..., wait=True)`, `get`, `list` |
| `client.eval_sets` | `import_file`, `import_text`, `version("halcyon-retrieval@1")`, `list` |
| `client.evals` | `run(run, retrieval=..., eval_set=..., split=..., wait=True)`, `get`, `list`, `compare(a, b)` |
| `client.experiments`, `client.profiles` | plan, create, start and follow experiments; export, import, list and download knowledge profiles |
| `client.tests` | `validate`, `save`, `run(suite or source=..., trigger="ci", wait=True)`, `get`, `list`, `history(name)`; a `TestRun` has `.passed`, `.checks()` and `.report("junit" / "json" / "markdown")` |
| `client.reports` | `table(rows)`: a Markdown table |

- **References:** `name` (the latest version), `name@latest`, `name@N`, or an ID.
- **Waiting:** `wait=True` (or `.wait()`) follows the run's server-sent events with backoff and
  re-reads the record, since events are hints; an API without events is polled.
- **Results as data:** lists are `Records`, an eval run has `Metrics`, a comparison is a
  `Comparison`; each has `.to_dicts()` and `.to_frame()` (pandas: `pip install 'stratafi[frames]'`).
- **Errors:** `StratafiError` and its kinds (`ConfigError`, `ApiError`, `UnreachableError`,
  `WaitTimeout`), each with the command line's exit code. Messages carry the API's problem
  details, which never echo what was sent.

## The command

```bash
stratafi corpus create halcyon
stratafi corpus upload halcyon --manifest corpus/manifest.yaml
stratafi corpus freeze halcyon
stratafi pipeline validate pipeline.yaml
stratafi pipeline push pipeline.yaml
stratafi retrieval push retrieval.yaml
stratafi run start --pipeline baseline-v1 --corpus halcyon@latest --until chunk --sample 20 --wait
stratafi evalset import retrieval.yaml --name halcyon-retrieval --split-test 0.3
stratafi eval run --run RUN_ID --retrieval hybrid-v1 --eval-set halcyon-retrieval@1 --split dev --wait
stratafi compare EVAL_RUN_A EVAL_RUN_B --format markdown
stratafi experiment plan experiment.yaml
stratafi experiment start experiment.yaml --confirm --wait
stratafi profile export TRIAL_ID --experiment EXPERIMENT_ID --name kp-policies
stratafi test validate suite.yaml
stratafi test run suite.yaml --junit results.xml --json results.json --markdown summary.md
stratafi test report TEST_RUN_ID --output junit
stratafi eval search "metrics.recall@5 > 0.8 and params.rerank.component = 'rerank/none'" --order-by "metrics.mrr@10 DESC"
stratafi eval tag EVAL_RUN_ID team=rag --title "Hybrid, no reranker"
stratafi eval note EVAL_RUN_ID --file why.md
stratafi eval compare EVAL_RUN_A EVAL_RUN_B EVAL_RUN_C --metric recall@5
stratafi eval trace EVAL_RUN_ID ITEM_ID
stratafi model compare --eval-set halcyon-retrieval@1 --role rerank
stratafi model alias MODEL_ID champion
stratafi map --eval-run EVAL_RUN_ID > pipeline.mmd
```

Tracking and comparison in the client
([the guide](../../docs/guides/tracking-and-comparison.md)): `client.evals.search(filter,
order_by=..., columns=...)` takes the search language, MLflow's `search_runs` filter;
`annotate(title=, tags=, note=)`, `archive()` and `restore()` on runs, eval runs and experiments;
`client.evals.compare_many(runs, baseline=...)` pairs two to six eval runs; an eval run's
`trace(item_id)`, `latency()` and `map()`; `client.models.compare(eval_set, role="rerank")`,
`set_alias`, `remove_alias` and `by_alias` on models and profiles; and `client.maps` for the map
of any saved spec and config, and every RAG pipeline the workspace has evaluated.

`--format json` prints the API's records. Exit codes: `0` passed, `1` a check failed (an invalid
spec, a run or eval run that didn't succeed, a test run with a failed or errored check), `2` a
usage or configuration error (a test run whose suite names something that doesn't exist), `3` a
platform error (unreachable, out of quota, a server error, a wait that timed out).

## Test suites in CI

`stratafi test run suite.yaml` sends the suite file (saved as its name's next version when it
changed), waits for the run, writes the reports it's asked for, and exits by the outcome.
`examples/ci/github-actions.yml` is a workflow for a repository that keeps suites beside its
pipeline specs: it installs the SDK, runs the suite with `--trigger ci` and a token
from the repository's secrets, uploads the JUnit XML, the record and the Markdown summary,
writes the summary to the job's page, and fails the job on a non-zero exit.
`examples/suites/policies-regression.yaml` is a suite to start from.

A receiver of the lab's webhooks checks each delivery with
`stratafi.webhooks.verify(secret, body, headers["Stratafi-Signature"])`.

## Custom checks

A suite's custom check is a Python function that takes a `stratafi.checks.CheckContext` and
returns a `CheckResult`; the suite names it as `python: checks/tables.py:tables_not_split`,
and `stratafi test run` sends that file, from beside the suite file, with the suite. The lab
runs it only in its sandbox (a container with no network, no credentials, the run's data
read-only and limits on time, memory, processes and output), and only in a workspace whose
owner turned custom checks on once the sandbox's security review was signed off. Elsewhere the
check ends as an error, `custom_checks_disabled`. Test a check on your machine first with
`CheckContext.from_records(directory, chunks=[...], documents=[...], queries=[...])`; its
`CheckResult.to_dict()` is what the lab keeps: IDs, codes and numbers, never text.
`examples/suites/tables.yaml` and its `checks/tables.py` are a suite of custom checks to start
from.

The [local development guide](../../docs/guides/local-development.md#the-command-line-and-the-sdk)
loads the Halcyon demo corpus with these commands.
