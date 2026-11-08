# Tracking and comparison

How to find, annotate, compare and map RAG pipelines in the lab. StrataFi offers MLflow's
tracking features on its own records, with the statistics its eval runs already carry. The
[architecture](../architecture.md#evaluation-and-comparison) explains what an eval run measures.

## Find runs: the search language

The Evals screen's table, `POST /api/v1/eval-runs/search`, `client.evals.search` and
`stratafi eval search` all take MLflow's `search_runs` filter: clauses joined by `and`.

```text
metrics.recall@5 > 0.8 and params.rerank.component = 'rerank/none' and tags.team = 'rag'
```

| Entity | Keys | Examples |
| --- | --- | --- |
| `metrics.` | `<metric>@<k>` (the `knowledge-hit-v1` rule, final list); `<checkpoint>.<metric>@<k>`; `<rule>.<checkpoint>.<metric>@<k>`; `latency.<step>.<p50, p95 or mean>`; `uplift.<metric>@<k>`; `findability`; `violations.avoid`; `lost.retrieval` | `metrics.fused.recall@all`, `metrics.latency.total_ms.p95 < 50` |
| `params.` | Dotted paths into the pipeline spec's stages and the retrieval config; list items by index; `models.<role>.<field>` | `params.chunk.params.max_tokens = 512`, `params.clean.0.component`, `params.models.embed.revision` |
| `tags.` | Your tags | `tags.team = 'search'` |
| `attributes.` | `status`, `split`, `title`, `eval_set`, `config`, `pipeline`, `corpus`, `experiment`, `created_at`, `archived` and more | `attributes.created_at > '2026-11-01'` |

Operators: `=`, `!=`, `<`, `<=`, `>`, `>=`, `LIKE`, `ILIKE`, `NOT LIKE` and `NOT ILIKE` (`%`,
`_`), `IN (…)`, `NOT IN (…)`, `IS NULL`, `IS NOT NULL`. Quote strings; put a key with other
characters in backticks (``params.`parse.route.pdf[scanned].component` ``). A clause on a key a
run doesn't have is false, as in SQL, except `IS NULL`. Order with keys and `ASC` or `DESC`:
`metrics.mrr@10 DESC`.

The table adds any metric, parameter or tag as a column, groups rows by a column with each
group's mean and best recall, and plots any two keys of the matches.

## Name and explain runs

Pipeline runs, eval runs and experiments have a title, tags (`key=value`) and a Markdown note,
and can be archived: hidden from lists and searches, never deleted. They're your text: StrataFi
never logs or audits them. `stratafi eval tag ID team=rag --title "…"` and
`stratafi eval note ID --file why.md` do it from the command line.

## Compare RAG pipelines

Select two to six eval runs of one eval set version and choose **Compare**
(`/compare/<id>/<id>/…`, `client.evals.compare_many`, `stratafi eval compare`). Every number has
its 95% interval, and differences are paired on the queries every run scored:

- **Overview:** each metric's interval per run; each run against the baseline, adjusted with
  Holm's method; and its group against the best run (best, tied, worse).
- **Charts:** a metric at every k, the share of queries whose evidence reaches each list (dense,
  keyword, fused, candidates, reranked, final), where queries were lost, quality against p95
  latency with its frontier, parallel coordinates of what differs, and slices by file type or
  tag.
- **Parameters**, **Queries** (a heatmap, and one query's final lists side by side),
  **Latency** (per step, with box plots) and **Maps** (each run's map, with what differs from the
  baseline marked).

Each run keeps one colour and shape in every chart. Fewer than 50 paired queries make the
intervals wide, and the screen says so.

## Trace a query

A query's page shows its trace: retrieval's steps on a timeline (keyword search, the query's
embedding, dense search, fusion and filtering, reranking), each with its lists and where the
evidence is in them, and the first list that dropped it. An eval run's page shows each step's
latency distribution and its slowest queries. `stratafi eval trace ID ITEM` prints the timeline.

## Compare models

The Models screen (`client.models.compare`, `stratafi model compare`) ranks embedders, rerankers
and adapters by their best eval run, then shows **controlled pairs**: eval runs that differ only
in the model, whose paired difference is the model's alone. A model can lead on its best run and
still not beat another when only the model changes. Aliases such as `champion` name one version
of a trained model (`stratafi model alias ID champion`) or, for owners, of a knowledge profile
(`stratafi profile alias`).

## Map a pipeline

The Maps screen lists every pipeline spec and retrieval config the workspace has evaluated, with
the services it calls, the datastores it reads and writes, and where text leaves the lab. A map
draws each stage, query step, datastore and service; select a step to see its reads, writes and
calls. A run's or eval run's map adds its numbers (tasks, cache hits, time, bytes; latency and
how often the evidence reaches each step). Maps name stores and services, never their addresses.
Download a map as SVG, copy it as Mermaid, or print it: `stratafi map --eval-run ID`.
