"""Sweep the chunker's size and compare retrieval: an SDK example.

For each size, the pipeline runs with ``chunk`` set to it (and its minimum to half of it), the
run is evaluated, and the eval set's headline metrics are tabulated.

    export STRATAFI_URL=http://127.0.0.1:8000
    uv run python examples/chunk_size_sweep.py --sizes 256,384,512,800

It needs a pipeline spec, a frozen corpus, a retrieval config and an eval set, by default the
ones ``scripts/local.py seed`` saves for the Halcyon demo corpus. With neither
``STRATAFI_TOKEN`` nor ``STRATAFI_DEV_SIGN_IN`` set, it signs in as the lab's local owner. Each
size's spec is saved as a variant of the pipeline
(``local-baseline.chunk.max_tokens-256.min_tokens-128``), so a second sweep reuses the same
specs and the stages it already ran.
"""

import argparse
import sys

from stratafi import Client, StratafiError


def sizes(text: str) -> list[int]:
    try:
        values = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("sizes are integers, such as 256,384,512") from None
    if not values or min(values) < 16:
        raise argparse.ArgumentTypeError("give at least one size, each 16 tokens or more")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep the chunk size; tabulate the metrics.")
    parser.add_argument("--pipeline", default="local-baseline")
    parser.add_argument("--corpus", default="halcyon@latest")
    parser.add_argument("--retrieval", default="local-hybrid")
    parser.add_argument("--eval-set", default="halcyon-retrieval@1")
    parser.add_argument("--split", default="dev", choices=("dev", "test", "all"))
    parser.add_argument("--sizes", type=sizes, default=[256, 384, 512, 800])
    args = parser.parse_args(argv)

    try:
        with Client.from_env() as client:
            base = client.pipelines.get(args.pipeline)
            rows = []
            for max_tokens in args.sizes:
                spec = base.with_params("chunk", max_tokens=max_tokens, min_tokens=max_tokens // 2)
                run = client.runs.start(spec, corpus=args.corpus, wait=True)
                if not run.succeeded:
                    print(f"the run for {max_tokens} tokens {run.status}", file=sys.stderr)
                    return 1
                result = client.evals.run(
                    run,
                    retrieval=args.retrieval,
                    eval_set=args.eval_set,
                    split=args.split,
                    wait=True,
                )
                if not result.succeeded:
                    print(f"the eval for {max_tokens} tokens {result.status}", file=sys.stderr)
                    return 1
                rows.append(
                    {
                        "max_tokens": max_tokens,
                        **result.metrics.select("recall@5", "mrr@10"),
                        "eval_run": result.id,
                    }
                )
            print(client.reports.table(rows))
    except StratafiError as error:
        print(f"chunk_size_sweep: {error}", file=sys.stderr)
        return error.exit_code
    return 0


if __name__ == "__main__":
    sys.exit(main())
