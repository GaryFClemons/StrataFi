# stratafi-worker

Claims jobs from the Postgres queue and runs them: pipeline runs and their stage tasks, eval
runs and their shards, experiments, test runs, answer runs, drafted questions, training sets
and training runs, and webhook deliveries. It also runs the retention sweep.

```bash
uv run --env-file .env stratafi-worker
```

`worker.slots` sets how many jobs of each resource class it runs at once, by default
`{cpu: 2, gpu: 1, io: 4}`. Each claim takes a lease that a heartbeat renews; a job whose worker
dies returns to the queue, and failures retry with backoff. SIGTERM or Ctrl+C stops it gracefully:
it claims nothing new, gives running jobs `worker.shutdown_grace_seconds`, then hands them back
without using up an attempt.

Parsers run in a sandboxed subprocess per document. On Linux they're confined, and the worker
makes itself undumpable, so a parser can't read its memory or environment. Custom checks run in Docker
containers, one per check, when the sandbox is configured
([the guide](../../docs/guides/local-development.md#custom-checks)).
[The architecture](../../docs/architecture.md#the-job-queue) describes the queue.
