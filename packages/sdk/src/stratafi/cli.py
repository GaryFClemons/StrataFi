"""The ``stratafi`` command line.

    stratafi corpus create halcyon
    stratafi corpus upload halcyon --manifest corpus/manifest.yaml
    stratafi corpus freeze halcyon
    stratafi pipeline validate pipeline.yaml
    stratafi pipeline push pipeline.yaml
    stratafi run start --pipeline baseline-v1 --corpus halcyon@latest --wait
    stratafi evalset import retrieval.yaml --name halcyon-retrieval --split-test 0.3
    stratafi eval run --run RUN_ID --retrieval hybrid-v1 --eval-set halcyon-retrieval@1 --wait
    stratafi compare EVAL_RUN_A EVAL_RUN_B --format markdown
    stratafi experiment plan experiment.yaml
    stratafi experiment start experiment.yaml --confirm --wait
    stratafi profile export TRIAL_ID --experiment EXPERIMENT_ID --name kp-policies
    stratafi test run suite.yaml --junit results.xml --json results.json --markdown summary.md
    stratafi train mine EVAL_RUN_ID --wait
    stratafi train start TRAINING_SET_ID --name halcyon-adapter --param epochs=40 --wait
    stratafi model download MODEL_ID --output halcyon-adapter.zip
    stratafi eval search "metrics.recall@5 > 0.8 and params.rerank.component = 'rerank/none'"
    stratafi eval tag EVAL_RUN_ID team=rag --title "Hybrid, no reranker"
    stratafi eval compare EVAL_RUN_A EVAL_RUN_B EVAL_RUN_C
    stratafi eval trace EVAL_RUN_ID ITEM_ID
    stratafi model compare --eval-set halcyon-retrieval@1 --role rerank
    stratafi model alias MODEL_ID champion
    stratafi map --eval-run EVAL_RUN_ID > pipeline.mmd

It's configured as ``Client.from_env`` is: ``STRATAFI_URL``, and ``STRATAFI_TOKEN`` (an API
token or an identity token) or ``STRATAFI_DEV_SIGN_IN``; ``--url`` overrides the address.
Results go to standard output (``--format json`` prints the API's records); progress and
errors go to standard error.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Callable
from datetime import date
from typing import Any

import yaml

from stratafi.client import Client
from stratafi.errors import CHECK_FAILED, PASSED, PLATFORM, USAGE, ConfigError, StratafiError
from stratafi.resources import EvalRun, Experiment, Run, TestRun, TrainingRun, TrainingSet
from stratafi.results import DEFAULT_CHECKPOINT, DEFAULT_RULE, markdown_table, plain
from stratafi.version import __version__

STAGES = ("extract", "scan", "parse", "clean", "chunk", "enrich", "embed", "index")
RULES = ("knowledge-hit-v1", "evidence-coverage-v1", "doc-v1")
CHECKPOINTS = ("dense", "keyword", "fused", "candidates", "reranked", "final")
SPLITS = ("dev", "test", "all")
FORMATS = ("text", "markdown", "json")
EXIT_CODES = """exit codes:
  0  passed
  1  a check failed: an invalid spec, or a run or eval run waited for that didn't succeed
  2  a usage or configuration error, such as a name that doesn't exist or a refused token
  3  a platform error: unreachable, out of quota, a server error, or a wait that timed out"""

Handler = Callable[[Client, argparse.Namespace], int]


def emit(args: argparse.Namespace, record: Any, text: str) -> None:
    """A result on standard output: ``text``, or the record as JSON."""
    if args.format == "json":
        print(json.dumps(plain(record), indent=2, ensure_ascii=False))
    else:
        print(text)


def say(text: str) -> None:
    """Progress on standard error."""
    print(text, file=sys.stderr, flush=True)


def progress(label: str) -> Callable[[dict[str, Any]], None]:
    """An ``on_update`` that reports each status change."""
    last: list[Any] = [None]

    def update(data: dict[str, Any]) -> None:
        status = data.get("status")
        if status != last[0]:
            last[0] = status
            say(f"{label}: {status}")

    return update


def _counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{value} {key}" for key, value in sorted(counts.items()))


def describe_run(run: Run) -> str:
    record = run.record
    text = f"run {run.id} {run.status}"
    if record.counts.tasks:
        text += f" (tasks: {_counts(record.counts.tasks)})"
    if record.error_code:
        text += f" [{record.error_code}]"
    return text


def describe_eval(run: EvalRun, rule: str = DEFAULT_RULE) -> str:
    record = run.record
    text = f"eval run {run.id} {run.status}"
    if record.results_cached:
        text += " (an identical run's results reused)"
    if record.error_code:
        text += f" [{record.error_code}]"
    metrics = run.metrics
    if metrics:
        text += f"\n\n{rule}, final list:\n\n{metrics.to_markdown(rule=rule)}"
    return text


def finish(args: argparse.Namespace, run: Run | EvalRun) -> int:
    """A finished run's result: passed when it succeeded."""
    if isinstance(run, EvalRun):
        emit(args, run, describe_eval(run, getattr(args, "rule", DEFAULT_RULE)))
    else:
        emit(args, run, describe_run(run))
    return PASSED if run.succeeded else CHECK_FAILED


# --- corpus -------------------------------------------------------------------------------


def corpus_create(client: Client, args: argparse.Namespace) -> int:
    corpus = client.corpora.create(args.name, description=args.description)
    emit(args, corpus, f"created corpus {corpus.name} ({corpus.id})")
    return PASSED


def corpus_upload(client: Client, args: argparse.Namespace) -> int:
    results = client.corpora.upload(
        args.name, args.paths, manifest=args.manifest, batch_size=args.batch_size
    )
    counts: Counter[str] = Counter()
    for result in results:
        counts.update(result.counts)
    files = sum(len(result.files) for result in results)
    added = sum(result.added_to_corpus for result in results)
    text = f"uploaded {files} files into {args.name}'s working set: {added} added"
    if counts:
        text += f" ({_counts(dict(counts))})"
    emit(args, results, text)
    return PASSED


def corpus_freeze(client: Client, args: argparse.Namespace) -> int:
    version = client.corpora.freeze(args.name)
    emit(
        args,
        version,
        f"{args.name} version {version.version} ({version.id}): {version.document_count} "
        f"documents, {version.total_bytes} bytes",
    )
    return PASSED


def corpus_list(client: Client, args: argparse.Namespace) -> int:
    corpora = client.corpora.list()
    rows = [[corpus.name, str(corpus.id), corpus.updated_at.isoformat()] for corpus in corpora]
    emit(args, corpora, markdown_table(["name", "id", "updated"], rows) if rows else "no corpora")
    return PASSED


def corpus_show(client: Client, args: argparse.Namespace) -> int:
    corpus = client.corpora.get(args.name)
    latest = corpus.latest_version
    frozen = (
        f"version {latest.version} has {latest.document_count} documents"
        if latest
        else "no frozen version"
    )
    emit(
        args,
        corpus,
        f"{corpus.name} ({corpus.id}): {corpus.entries} entries, {corpus.total_bytes} bytes; "
        f"{frozen}",
    )
    return PASSED


# --- pipeline and retrieval ---------------------------------------------------------------


def spec_validate(resource: str) -> Handler:
    def handler(client: Client, args: argparse.Namespace) -> int:
        verdict = getattr(client, resource).validate(args.file)
        if verdict.valid:
            emit(args, verdict, f"valid (spec hash {(verdict.spec_hash or '')[:12]})")
            return PASSED
        lines = ["invalid:", *(f"  {issue.path}: {issue.code}" for issue in verdict.issues)]
        emit(args, verdict, "\n".join(lines))
        return CHECK_FAILED

    return handler


def spec_push(resource: str) -> Handler:
    def handler(client: Client, args: argparse.Namespace) -> int:
        spec = getattr(client, resource).push(args.file)
        unchanged = "" if spec.created else ", unchanged"
        emit(args, spec.record, f"{spec.name} version {spec.version} ({spec.id}){unchanged}")
        return PASSED

    return handler


def spec_show(resource: str) -> Handler:
    def handler(client: Client, args: argparse.Namespace) -> int:
        spec = getattr(client, resource).get(args.ref)
        text = f"# {spec.name} version {spec.version} ({spec.id})\n" + yaml.safe_dump(
            spec.document, sort_keys=False, allow_unicode=True
        )
        emit(args, spec.record, text.rstrip("\n"))
        return PASSED

    return handler


def spec_list(resource: str) -> Handler:
    def handler(client: Client, args: argparse.Namespace) -> int:
        specs = getattr(client, resource).list()
        rows = [[spec.name, str(spec.version), str(spec.id)] for spec in specs]
        text = markdown_table(["name", "latest version", "id"], rows) if rows else "none saved"
        emit(args, [spec.record for spec in specs], text)
        return PASSED

    return handler


# --- run ----------------------------------------------------------------------------------


def run_start(client: Client, args: argparse.Namespace) -> int:
    sample = None
    if args.sample is not None:
        sample = {"size": args.sample, "seed": args.sample_seed}
    run = client.runs.start(
        args.pipeline,
        corpus=args.corpus,
        until=args.until,
        sample=sample,
        mode="interactive" if args.interactive else "batch",
    )
    if not args.wait:
        emit(args, run, describe_run(run))
        return PASSED
    say(f"run {run.id}: {run.status}")
    return finish(args, run.wait(timeout=args.timeout, on_update=progress(f"run {run.id}")))


def run_show(client: Client, args: argparse.Namespace) -> int:
    run = client.runs.get(args.run_id)
    emit(args, run, describe_run(run))
    return PASSED


def run_wait(client: Client, args: argparse.Namespace) -> int:
    run = client.runs.get(args.run_id)
    return finish(args, run.wait(timeout=args.timeout, on_update=progress(f"run {run.id}")))


def run_cancel(client: Client, args: argparse.Namespace) -> int:
    run = client.runs.get(args.run_id).cancel()
    emit(args, run, describe_run(run))
    return PASSED


# --- evalset ------------------------------------------------------------------------------


def evalset_import(client: Client, args: argparse.Namespace) -> int:
    options: dict[str, Any] = {}
    if args.input_format:
        options["format"] = args.input_format
    imported = client.eval_sets.import_file(
        args.file,
        name=args.name,
        description=args.description,
        as_of=args.as_of,
        k=args.k,
        test_fraction=args.split_test,
        seed=args.seed,
        resplit=args.resplit,
        freeze=not args.no_freeze,
        **options,
    )
    version = imported.version
    unchanged = "" if imported.created else ", unchanged"
    emit(
        args,
        imported,
        f"{imported.eval_set.name} version {version.version} ({version.id}): "
        f"{version.item_count} items ({_counts(version.split_counts)}){unchanged}",
    )
    return PASSED


def evalset_show(client: Client, args: argparse.Namespace) -> int:
    version = client.eval_sets.version(args.ref)
    state = "frozen" if version.frozen else "not frozen"
    if version.test_locked:
        state += f", test split locked ({version.test_reads} reads)"
    emit(
        args,
        version,
        f"version {version.version} ({version.id}): {version.item_count} items "
        f"({_counts(version.split_counts)}); {state}",
    )
    return PASSED


def evalset_list(client: Client, args: argparse.Namespace) -> int:
    sets = client.eval_sets.list()
    rows = [
        [
            found.name,
            str(found.latest_version.version) if found.latest_version else "-",
            str(found.latest_version.item_count) if found.latest_version else "-",
            str(found.id),
        ]
        for found in sets
    ]
    text = markdown_table(["name", "latest version", "items", "id"], rows) if rows else "none"
    emit(args, sets, text)
    return PASSED


# --- eval ---------------------------------------------------------------------------------


def eval_run(client: Client, args: argparse.Namespace) -> int:
    options = {
        key: value
        for key, value in (
            ("seed", args.seed),
            ("resamples", args.resamples),
            ("coverage_threshold", args.coverage_threshold),
            ("shard_size", args.shard_size),
        )
        if value is not None
    }
    started = client.evals.run(
        args.run_id,
        retrieval=args.retrieval,
        eval_set=args.eval_set,
        split=args.split,
        confirm_test_read=args.confirm_test_read,
        options=options,
    )
    if not args.wait:
        emit(args, started, describe_eval(started, args.rule))
        return PASSED
    say(f"eval run {started.id}: {started.status}")
    label = f"eval run {started.id}"
    return finish(args, started.wait(timeout=args.timeout, on_update=progress(label)))


def eval_show(client: Client, args: argparse.Namespace) -> int:
    found = client.evals.get(args.eval_run_id)
    emit(args, found, describe_eval(found, args.rule))
    return PASSED


def eval_wait(client: Client, args: argparse.Namespace) -> int:
    found = client.evals.get(args.eval_run_id)
    label = f"eval run {found.id}"
    return finish(args, found.wait(timeout=args.timeout, on_update=progress(label)))


def eval_cancel(client: Client, args: argparse.Namespace) -> int:
    found = client.evals.get(args.eval_run_id).cancel()
    emit(args, found, describe_eval(found, args.rule))
    return PASSED


# --- experiment ---------------------------------------------------------------------------


def describe_plan(plan: dict[str, Any] | None) -> str:
    """A plan as Markdown: trials, the unique work per stage, the estimate and the budget."""
    if not plan:
        return "no plan"
    lines = [
        f"{plan['trials']} trials ({plan['strategy']}; {plan['grid']} combinations, "
        f"{plan['excluded']} excluded) over {plan['work']['pipelines']} pipeline runs",
        "",
        "| Stage | Every trial alone | Unique | Cached | To run |",
        "| --- | --- | --- | --- | --- |",
    ]
    for stage, work in plan["work"]["stages"].items():
        lines.append(
            f"| {stage} | {work['tasks']} | {work['unique']} | {work['cached']} | "
            f"{work['to_run']} |"
        )
    search = plan.get("search")
    if search:
        rungs = " -> ".join(f"{rung['queries']} queries" for rung in search["rungs"])
        lines += [
            "",
            f"Successive halving (eta {search['eta']}): {rungs}; nominally "
            f"{search['share']:.0%} of the grid's queries.",
        ]
    estimate = plan["estimate"]
    lines += [
        "",
        f"Estimate: {estimate['wall_seconds']:g} s, ${estimate['cost_usd']['total']:g}; "
        f"{estimate['eval']['to_run']} of {estimate['eval']['trials']} trials to evaluate.",
    ]
    over = [name for name, check in plan["budget"].items() if check.get("over")]
    if over:
        lines.append(f"Over budget: {', '.join(over)}.")
    if plan.get("warnings"):
        lines.append(f"Warnings: {', '.join(plan['warnings'])}.")
    return "\n".join(lines)


def describe_experiment(experiment: Experiment) -> str:
    record = experiment.record
    text = f"experiment {experiment.id} {experiment.status}"
    trials = (record.counts or {}).get("trials") or {}
    if trials:
        text += f" (trials: {_counts(trials)})"
    if record.stop_reason:
        text += f" [stopped by {record.stop_reason}]"
    spent = record.spent or {}
    if spent:
        text += f", spent ${spent.get('total_usd', 0):g}"
    return text


def experiment_plan(client: Client, args: argparse.Namespace) -> int:
    planned = client.experiments.plan(args.file)
    if not planned.valid:
        for issue in planned.issues:
            say(f"{issue.path}: {issue.code}")
        emit(args, planned, "invalid")
        return CHECK_FAILED
    emit(args, planned, describe_plan(planned.plan))
    return PASSED


def experiment_start(client: Client, args: argparse.Namespace) -> int:
    if not args.confirm:
        say("stratafi: review the plan (stratafi experiment plan), then start with --confirm")
        return USAGE
    created = client.experiments.create(args.file)
    say(f"experiment {created.id}: planned")
    started = created.start()
    if not args.wait:
        emit(args, started, describe_experiment(started))
        return PASSED
    label = f"experiment {started.id}"
    done = started.wait(timeout=args.timeout, on_update=progress(label))
    emit(args, done, describe_experiment(done))
    return PASSED if done.status in ("succeeded", "stopped") else CHECK_FAILED


def experiment_show(client: Client, args: argparse.Namespace) -> int:
    found = client.experiments.get(args.experiment_id)
    emit(args, found, describe_experiment(found))
    return PASSED


def experiment_wait(client: Client, args: argparse.Namespace) -> int:
    found = client.experiments.get(args.experiment_id)
    done = found.wait(timeout=args.timeout, on_update=progress(f"experiment {found.id}"))
    emit(args, done, describe_experiment(done))
    return PASSED if done.status in ("succeeded", "stopped") else CHECK_FAILED


def experiment_cancel(client: Client, args: argparse.Namespace) -> int:
    found = client.experiments.get(args.experiment_id).cancel()
    emit(args, found, describe_experiment(found))
    return PASSED


# --- profile ------------------------------------------------------------------------------


def describe_profile(profile: Any) -> str:
    text = f"profile {profile.name} v{profile.version} {profile.status}"
    if profile.created is False:
        text += " (the latest version holds this profile already)"
    return text


def profile_export(client: Client, args: argparse.Namespace) -> int:
    exported = client.profiles.export(
        args.trial_id,
        experiment=args.experiment,
        name=args.name,
        description=args.description,
    )
    emit(args, exported, describe_profile(exported))
    return PASSED


def profile_download(client: Client, args: argparse.Namespace) -> int:
    print(client.profiles.download(args.profile_id, format=args.output), end="")
    return PASSED


def profile_import(client: Client, args: argparse.Namespace) -> int:
    imported = client.profiles.import_file(args.file)
    emit(args, imported, describe_profile(imported))
    return PASSED


def compare(client: Client, args: argparse.Namespace) -> int:
    comparison = client.evals.compare(
        args.a,
        args.b,
        rule=args.rule,
        checkpoint=args.checkpoint,
        metrics=args.metric,
        resamples=args.resamples,
        seed=args.seed,
    )
    emit(args, comparison, comparison.to_markdown())
    return PASSED


# --- tracking and comparison --------------------------------------------------------------


MISSING = "-"


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return MISSING
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def eval_search(client: Client, args: argparse.Namespace) -> int:
    found = client.evals.search(
        args.filter or "",
        order_by=args.order_by or (),
        columns=args.column or (),
        include_archived=args.include_archived,
        max_results=args.max_results,
    )
    columns = list(args.column or [])
    rows = []
    for item in found:
        headline = item.headline
        recall = next((v for k, v in headline.items() if k.startswith("recall@")), None)
        rows.append(
            [
                str(item.id),
                item.label,
                item.status,
                item.split,
                _fmt(recall),
                _fmt(headline.get("mrr@10")),
                _fmt(item.latency_p95_ms, 1),
                ", ".join(f"{key}={value}" for key, value in sorted(item.tags.items())),
                *[_fmt(item.values.get(column)) for column in columns],
            ]
        )
    header = ["eval run", "label", "status", "split", "recall@k", "mrr@10", "p95 ms", "tags"]
    emit(args, found, markdown_table([*header, *columns], rows) if rows else "no eval runs match")
    return PASSED


def _annotated(client: Client, args: argparse.Namespace) -> EvalRun:
    return client.evals.get(args.eval_run_id)


def _describe_annotation(eval_run_id: str, annotation: Any) -> str:
    tags = ", ".join(f"{key}={value}" for key, value in sorted(annotation.tags.items()))
    state = "archived" if annotation.archived else "active"
    title = annotation.title or "(no title)"
    return f"eval run {eval_run_id}: {title}; {state}; tags: {tags or 'none'}"


def eval_tag(client: Client, args: argparse.Namespace) -> int:
    tags = {key: str(value) for key, value in _params(args.pairs).items()}
    found = _annotated(client, args).annotate(
        title=args.title, tags=tags, delete_tags=args.delete or ()
    )
    emit(args, found, _describe_annotation(args.eval_run_id, found))
    return PASSED


def eval_note(client: Client, args: argparse.Namespace) -> int:
    if args.clear:
        note = ""
    else:
        try:
            with open(args.file, encoding="utf-8") as handle:
                note = handle.read()
        except (OSError, UnicodeDecodeError):
            raise ConfigError(f"can't read {args.file}") from None
    found = _annotated(client, args).annotate(note=note)
    emit(args, found, _describe_annotation(args.eval_run_id, found))
    return PASSED


def eval_archive(client: Client, args: argparse.Namespace) -> int:
    found = _annotated(client, args).annotate(archived=args.action == "archive")
    emit(args, found, _describe_annotation(args.eval_run_id, found))
    return PASSED


def eval_compare(client: Client, args: argparse.Namespace) -> int:
    found = client.evals.compare_many(
        args.eval_run_ids,
        baseline=args.baseline,
        rule=args.rule,
        checkpoint=args.checkpoint,
        metrics=args.metric,
        resamples=args.resamples,
        seed=args.seed,
    )
    labels = [f"{index + 1}. {run.label}" for index, run in enumerate(found.runs)]
    rows = []
    for row in found.table:
        cells = []
        for index, cell in enumerate(row.cells):
            text = f"{_fmt(cell.mean)} [{_fmt(cell.low)}, {_fmt(cell.high)}]"
            if index != found.baseline and cell.delta is not None:
                text += f" Δ {cell.delta:+.3f} (p {_fmt(cell.p_holm)})"
            if cell.group:
                text += f" {cell.group}"
            cells.append(text)
        rows.append([row.metric, *cells])
    lines = [
        f"{len(found.runs)} eval runs on {found.paired} paired queries; baseline: "
        f"{labels[found.baseline]}; {found.rule}, {found.checkpoint}",
        "",
        markdown_table(["metric", *labels], rows),
    ]
    if found.params:
        lines += [
            "",
            markdown_table(
                ["parameter", *labels],
                [[param.key, *[_fmt(value) for value in param.values]] for param in found.params],
            ),
        ]
    if found.warnings:
        lines += ["", "warnings: " + ", ".join(found.warnings)]
    emit(args, found, "\n".join(lines))
    return PASSED


def eval_trace(client: Client, args: argparse.Namespace) -> int:
    found = _annotated(client, args).trace(args.item_id, rule=args.rule)
    scale = 40 / found.total_ms if found.total_ms else 0
    rows = []
    for span in found.spans:
        bar = " " * round(span.start_ms * scale) + "█" * max(1, round(span.duration_ms * scale))
        lists = ", ".join(
            f"{item.checkpoint} {item.length}"
            + (f" (first relevant {item.first_relevant_rank})" if item.first_relevant_rank else "")
            for item in span.lists
        )
        rows.append([span.name, f"{span.start_ms:.1f}", f"{span.duration_ms:.1f}", bar, lists])
    text = markdown_table(["step", "start ms", "ms", "timeline", "lists"], rows)
    if found.lost_at_checkpoint:
        text += f"\n\nthe evidence was dropped at: {found.lost_at_checkpoint}"
    emit(args, found, text)
    return PASSED


def model_compare(client: Client, args: argparse.Namespace) -> int:
    found = client.models.compare(
        args.eval_set, role=args.role, metric=args.metric, split=args.split
    )
    rows = [
        [
            row.label,
            row.component,
            str(row.runs),
            _fmt((row.best or {}).get("mean")),
            f"[{_fmt((row.best or {}).get('low'))}, {_fmt((row.best or {}).get('high'))}]",
            _fmt(row.latency_p95_ms, 1),
            row.egress or MISSING,
        ]
        for row in found.models
    ]
    text = markdown_table(
        ["model", "component", "runs", f"best {found.metric}", "95% CI", "p95 ms", "egress"], rows
    )
    labels = {row.key: row.label for row in found.models}
    if found.pairs:
        pairs = []
        for pair in found.pairs:
            result = pair.result or {}
            pairs.append(
                [
                    labels.get(pair.a_model, pair.a_model),
                    labels.get(pair.b_model, pair.b_model),
                    _fmt(result.get("delta")),
                    f"[{_fmt(result.get('low'))}, {_fmt(result.get('high'))}]",
                    _fmt(result.get("p_value")),
                    str(result.get("n", 0)),
                ]
            )
        text += "\n\ncontrolled pairs (B minus A):\n\n" + markdown_table(
            ["A", "B", "delta", "95% CI", "p", "queries"], pairs
        )
    if found.warnings:
        text += "\n\nwarnings: " + ", ".join(found.warnings)
    emit(args, found, text)
    return PASSED


def model_alias(client: Client, args: argparse.Namespace) -> int:
    if args.delete:
        client.models.remove_alias(args.model_id, args.alias)
        emit(args, {"removed": args.alias}, f"removed {args.alias} from {args.model_id}")
        return PASSED
    found = client.models.set_alias(args.model_id, args.alias)
    moved = f", moved from {found.moved_from}" if found.moved_from else ""
    emit(args, found, f"{found.name}@{found.alias} is version {found.version}{moved}")
    return PASSED


def profile_alias(client: Client, args: argparse.Namespace) -> int:
    if args.delete:
        client.profiles.remove_alias(args.profile_id, args.alias)
        emit(args, {"removed": args.alias}, f"removed {args.alias} from {args.profile_id}")
        return PASSED
    found = client.profiles.set_alias(args.profile_id, args.alias)
    moved = f", moved from {found.moved_from}" if found.moved_from else ""
    emit(args, found, f"{found.name}@{found.alias} is version {found.version}{moved}")
    return PASSED


def pipeline_map(client: Client, args: argparse.Namespace) -> int:
    if args.eval_run:
        found = client.maps.of_eval_run(args.eval_run)
    elif args.run:
        found = client.maps.of_run(args.run)
    elif args.pipeline or args.retrieval:
        found = client.maps.get(pipeline=args.pipeline, retrieval=args.retrieval)
    else:
        raise ConfigError("name --pipeline, --retrieval, --run or --eval-run")
    if args.format == "json":
        emit(args, found, "")
    else:
        print(found.mermaid, end="")
    return PASSED


# --- arguments ----------------------------------------------------------------------------


def _number(kind: type, minimum: float, *, above: bool = False) -> Callable[[str], Any]:
    def parse(text: str) -> Any:
        try:
            value = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} isn't a number") from None
        if value < minimum or (above and value == minimum):
            bound = f"more than {minimum}" if above else f"at least {minimum}"
            raise argparse.ArgumentTypeError(f"{text} must be {bound}")
        return value

    return parse


def _fraction(text: str) -> float:
    value = _number(float, 0.0)(text)
    if value > 0.9:
        raise argparse.ArgumentTypeError(f"{text} must be at most 0.9")
    return value


def _day(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} isn't a date (YYYY-MM-DD)") from None


# --- training -----------------------------------------------------------------------------


def describe_training_set(training_set: TrainingSet) -> str:
    record = training_set.record
    stats = record.stats or {}
    lines = [f"training set {record.id} ({record.name}): {record.status}"]
    if record.error_code:
        lines.append(f"error: {record.error_code}")
    if stats:
        lines.append(
            f"{stats.get('records', 0)} records: {stats.get('pairs', 0)} with evidence, "
            f"{stats.get('abstentions', 0)} without; {stats.get('positives', 0)} positives, "
            f"{stats.get('negatives', 0)} negatives; {stats.get('validation', 0)} held out"
        )
        leak = stats.get("leak_guard") or {}
        lines.append(
            f"leak guard: {leak.get('near_duplicates', 0)} near duplicates of test queries dropped"
        )
    return "\n".join(lines)


def describe_training_run(run: TrainingRun) -> str:
    record = run.record
    lines = [f"training run {record.id} ({record.name}, {record.trainer}): {record.status}"]
    if record.error_code:
        lines.append(f"error: {record.error_code}")
    metrics = record.metrics or {}
    epochs = metrics.get("epochs") or []
    if epochs:
        lines.append("| epoch | train loss | recall@10 | mrr@10 |")
        lines.append("| --- | --- | --- | --- |")
        baseline = metrics.get("baseline") or {}
        if baseline:
            lines.append(f"| 0 (base) | | {baseline.get('recall@10')} | {baseline.get('mrr@10')} |")
        for epoch in epochs:
            mark = " (kept)" if epoch.get("epoch") == metrics.get("best_epoch") else ""
            lines.append(
                f"| {epoch.get('epoch')}{mark} | {epoch.get('train_loss')} | "
                f"{epoch.get('recall@10')} | {epoch.get('mrr@10')} |"
            )
    if record.model_id:
        cached = " (reused)" if record.model_cached else ""
        lines.append(f"model: {record.model_id}{cached}")
    return "\n".join(lines)


def describe_model(model: Any) -> str:
    use = (model.card or {}).get("use", {})
    lines = [
        f"model {model.id}: {model.name}@{model.version} ({model.kind}), {model.status}"
        + ("" if model.stored else ", purged"),
        f"base: {model.base.get('model')} ({model.base.get('revision')})",
    ]
    if use:
        lines.append(f"use: {use.get('how')}")
    return "\n".join(lines)


def _params(pairs: list[str] | None) -> dict[str, Any]:
    """``key=value`` pairs; values read as YAML, so numbers and booleans keep their types."""
    params: dict[str, Any] = {}
    for pair in pairs or []:
        key, separator, value = pair.partition("=")
        if not separator or not key:
            raise ConfigError(f"{pair!r} isn't key=value")
        params[key] = yaml.safe_load(value)
    return params


def train_mine(client: Client, args: argparse.Namespace) -> int:
    options = {
        key: value
        for key, value in (
            ("max_negatives", args.max_negatives),
            ("max_positives", args.max_positives),
            ("validation_share", args.validation_share),
            ("seed", args.seed),
        )
        if value is not None
    }
    started = client.training.mine(args.eval_run_id, name=args.name, options=options)
    if not args.wait:
        emit(args, started, describe_training_set(started))
        return PASSED
    done = started.wait(timeout=args.timeout)
    emit(args, done, describe_training_set(done))
    return PASSED if done.succeeded else CHECK_FAILED


def train_export(client: Client, args: argparse.Namespace) -> int:
    text = client.training.get_set(args.training_set_id).export(args.as_, split=args.split)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
        say(f"wrote {len(text.splitlines())} rows to {args.output}")
    else:
        print(text, end="")
    return PASSED


def train_start(client: Client, args: argparse.Namespace) -> int:
    started = client.training.train(
        args.training_set_id, name=args.name, trainer=args.trainer, params=_params(args.param)
    )
    if not args.wait:
        emit(args, started, describe_training_run(started))
        return PASSED
    say(f"training run {started.id}: {started.status}")
    label = f"training run {started.id}"
    done = started.wait(timeout=args.timeout, on_update=progress(label))
    emit(args, done, describe_training_run(done))
    return PASSED if done.succeeded else CHECK_FAILED


def train_show(client: Client, args: argparse.Namespace) -> int:
    run = client.training.get_run(args.training_run_id)
    emit(args, run, describe_training_run(run))
    return PASSED


def train_wait(client: Client, args: argparse.Namespace) -> int:
    run = client.training.get_run(args.training_run_id)
    done = run.wait(timeout=args.timeout, on_update=progress(f"training run {run.id}"))
    emit(args, done, describe_training_run(done))
    return PASSED if done.succeeded else CHECK_FAILED


def train_cancel(client: Client, args: argparse.Namespace) -> int:
    run = client.training.get_run(args.training_run_id).cancel()
    emit(args, run, describe_training_run(run))
    return PASSED


def model_list(client: Client, args: argparse.Namespace) -> int:
    found = client.models.list(kind=args.kind, name=args.name)
    rows = [
        [str(m.id), f"{m.name}@{m.version}", m.kind, m.status, str(m.base.get("model"))]
        for m in found
    ]
    emit(args, found, markdown_table(["id", "model", "kind", "status", "base"], rows))
    return PASSED


def model_show(client: Client, args: argparse.Namespace) -> int:
    model = client.models.get(args.model_id)
    emit(args, model, describe_model(model))
    return PASSED


def model_withdraw(client: Client, args: argparse.Namespace) -> int:
    model = client.models.withdraw(args.model_id)
    emit(args, model, describe_model(model))
    return PASSED


def model_download(client: Client, args: argparse.Namespace) -> int:
    path = client.models.download(args.model_id, args.output)
    say(f"wrote {path}")
    return PASSED


# --- test ---------------------------------------------------------------------------------

# A run that ended ``error`` for one of these reasons was misconfigured: a usage error (2).
CONFIG_ERRORS = ("_not_found", "_invalid", "_not_frozen", "_split_empty", "_not_succeeded")
SUITE_REF_RE = r"^[a-z0-9][a-z0-9._-]{0,63}(@([1-9][0-9]*|latest))?$"


def suite_exit_code(run: TestRun) -> int:
    """``passed`` 0; ``failed`` (a check failed or errored) 1; ``error`` 2 when the suite's
    references or configuration were wrong, else 3, as ``cancelled`` is."""
    if run.status == "passed":
        return PASSED
    if run.status == "failed":
        return CHECK_FAILED
    code = run.record.error_code or ""
    if run.status == "error" and code.endswith(CONFIG_ERRORS):
        return USAGE
    return PLATFORM


def describe_test_run(run: TestRun) -> str:
    record = run.record
    summary = record.summary or {}
    text = f"test run {run.id} of {record.suite_name} v{record.suite_version}: {run.status}"
    if record.error_code:
        text += f" ({record.error_code})"
    if summary:
        text += (
            f"\n{summary.get('checks', 0)} checks: {summary.get('passed', 0)} passed, "
            f"{summary.get('failed', 0)} failed, {summary.get('errored', 0)} errors, "
            f"{summary.get('skipped', 0)} skipped"
        )
    rows = [
        [
            check.id,
            check.type,
            check.result,
            "" if check.value is None else f"{check.value:.4g}",
            check.code or "",
        ]
        for check in record.checks or []
    ]
    if rows:
        text += "\n\n" + markdown_table(["check", "type", "result", "value", "code"], rows)
    return text


def _write(path: str | None, text: str) -> None:
    if path:
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
        except OSError:
            raise ConfigError(f"can't write {path}") from None
        say(f"wrote {path}")


def suite_run(client: Client, args: argparse.Namespace) -> int:
    if os.path.exists(args.suite):
        started = client.tests.run(source=args.suite, trigger=args.trigger)
    elif re.fullmatch(SUITE_REF_RE, args.suite):
        started = client.tests.run(args.suite, trigger=args.trigger)
    else:
        raise ConfigError(f"no suite file {args.suite}, and it isn't a suite's name")
    say(f"test run {started.id}: {started.status}")
    if args.no_wait:
        emit(args, started, describe_test_run(started))
        return PASSED
    done = started.wait(timeout=args.timeout, on_update=progress(f"test run {started.id}"))
    _write(args.junit, done.report("junit"))
    _write(args.json, done.report("json"))
    _write(args.markdown, done.report("markdown", link_base=args.link_base))
    emit(args, done, describe_test_run(done))
    return suite_exit_code(done)


def suite_validate(client: Client, args: argparse.Namespace) -> int:
    verdict = client.tests.validate(args.file)
    if verdict.valid:
        emit(args, verdict, "valid")
        return PASSED
    lines = ["invalid:", *(f"  {issue.path}: {issue.code}" for issue in verdict.issues)]
    emit(args, verdict, "\n".join(lines))
    return CHECK_FAILED


def suite_push(client: Client, args: argparse.Namespace) -> int:
    saved = client.tests.save(args.file)
    emit(args, saved, f"{saved.name} version {saved.version} ({saved.id})")
    return PASSED


def suite_show(client: Client, args: argparse.Namespace) -> int:
    found = client.tests.get(args.test_run_id)
    emit(args, found, describe_test_run(found))
    return PASSED


def suite_wait(client: Client, args: argparse.Namespace) -> int:
    found = client.tests.get(args.test_run_id)
    done = found.wait(timeout=args.timeout, on_update=progress(f"test run {found.id}"))
    emit(args, done, describe_test_run(done))
    return suite_exit_code(done)


def suite_report(client: Client, args: argparse.Namespace) -> int:
    found = client.tests.get(args.test_run_id)
    print(found.report(args.output, link_base=args.link_base), end="")
    return PASSED


def suite_list(client: Client, args: argparse.Namespace) -> int:
    found = client.tests.list(suite=args.suite)
    rows = [
        [str(run.id), run.suite_name, str(run.suite_version), run.status, run.trigger]
        for run in found
    ]
    emit(args, found, markdown_table(["test run", "suite", "version", "status", "trigger"], rows))
    return PASSED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stratafi",
        description="The Stratafi command line: corpora, pipeline specs, retrieval configs, "
        "runs, eval sets, eval runs, searches and comparisons, experiments, knowledge "
        "profiles, test suites, training sets, training runs and trained models, and maps.",
        epilog=EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"stratafi {__version__}")
    parser.add_argument("--url", help="the API's address (default: STRATAFI_URL)")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--format",
        choices=FORMATS,
        default="text",
        help="text or markdown (the same: tables are Markdown), or json (the API's records)",
    )
    waiting = argparse.ArgumentParser(add_help=False)
    waiting.add_argument(
        "--timeout", type=_number(float, 0.0, above=True), help="seconds to wait at most"
    )
    count = _number(int, 1)
    seed = _number(int, 0)

    def group(name: str, help: str) -> Any:
        command = commands.add_parser(name, help=help, description=help)
        return command.add_subparsers(dest="action", metavar="ACTION", required=True)

    def leaf(
        within: Any, name: str, handler: Handler, help: str, *, wait: bool = False
    ) -> argparse.ArgumentParser:
        parents = [shared, waiting] if wait else [shared]
        command = within.add_parser(name, parents=parents, help=help, description=help)
        command.set_defaults(handler=handler)
        return command

    corpus = group("corpus", "create corpora, upload into them, freeze them")
    command = leaf(corpus, "create", corpus_create, "create a corpus")
    command.add_argument("name")
    command.add_argument("--description", default="")
    command = leaf(corpus, "upload", corpus_upload, "upload files into a corpus's working set")
    command.add_argument("name")
    command.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help="a file (named by its file name) or a folder (each file named by its path in it)",
    )
    command.add_argument(
        "--manifest",
        help="a manifest.yaml (the corpus manifest format) that gives the entries their "
        "metadata; without PATHs, the files it lists are uploaded",
    )
    command.add_argument("--batch-size", type=count, default=100, help="files per request")
    command = leaf(corpus, "freeze", corpus_freeze, "freeze the working set as a version")
    command.add_argument("name")
    leaf(corpus, "list", corpus_list, "list the corpora")
    command = leaf(corpus, "show", corpus_show, "show a corpus and its latest version")
    command.add_argument("name")

    for noun, resource, what in (
        ("pipeline", "pipelines", "pipeline spec"),
        ("retrieval", "retrieval", "retrieval config"),
    ):
        specs = group(noun, f"validate, save and show {what}s")
        command = leaf(specs, "validate", spec_validate(resource), f"validate a {what} file")
        command.add_argument("file", help="YAML or JSON")
        command = leaf(specs, "push", spec_push(resource), f"save a {what} as a new version")
        command.add_argument("file", help="YAML or JSON")
        command = leaf(specs, "show", spec_show(resource), f"show a saved {what}")
        command.add_argument("ref", metavar="NAME", help="name (the latest version) or name@N")
        leaf(specs, "list", spec_list(resource), f"list the {what}s, latest versions")

    runs = group("run", "start, follow and cancel pipeline runs")
    command = leaf(runs, "start", run_start, "start a pipeline run", wait=True)
    command.add_argument("--pipeline", required=True, help="name, name@N, or a version's ID")
    command.add_argument("--corpus", required=True, help="name@latest, name@N, or a version's ID")
    command.add_argument("--until", choices=STAGES, help="stop after this stage")
    command.add_argument("--sample", type=count, help="run on this many of the corpus's entries")
    command.add_argument("--sample-seed", type=seed, help="the sample's seed (default 0)")
    command.add_argument(
        "--interactive", action="store_true", help="run ahead of batch work (the UI's mode)"
    )
    command.add_argument("--wait", action="store_true", help="wait until it finishes")
    for action, handler, help in (
        ("show", run_show, "show a run"),
        ("wait", run_wait, "wait until a run finishes"),
        ("cancel", run_cancel, "cancel a run"),
    ):
        command = leaf(runs, action, handler, help, wait=action == "wait")
        command.add_argument("run_id", metavar="RUN_ID")

    evalsets = group("evalset", "import and show eval sets")
    command = leaf(evalsets, "import", evalset_import, "import an eval set as a new version")
    command.add_argument("file", help="YAML, JSONL or CSV (by its extension)")
    command.add_argument("--name", required=True)
    command.add_argument(
        "--split-test",
        type=_fraction,
        default=0.3,
        help="the fraction of new items for the test split (default 0.3)",
    )
    command.add_argument("--seed", type=seed, default=0, help="the split's seed (default 0)")
    command.add_argument("--as-of", type=_day, help="the day the queries are asked (YYYY-MM-DD)")
    command.add_argument("--k", type=count, help="the cutoff its rule uses (default: the set's)")
    command.add_argument("--input-format", choices=("yaml", "jsonl", "csv"))
    command.add_argument("--description")
    command.add_argument(
        "--resplit", action="store_true", help="split again items that have a split"
    )
    command.add_argument("--no-freeze", action="store_true", help="leave the version editable")
    command = leaf(evalsets, "show", evalset_show, "show an eval set version")
    command.add_argument("ref", metavar="NAME@N", help="name@N, or name@latest")
    leaf(evalsets, "list", evalset_list, "list the eval sets")

    evals = group("eval", "run, follow and cancel eval runs")
    command = leaf(evals, "run", eval_run, "evaluate a pipeline run's indexes", wait=True)
    command.add_argument("--run", required=True, dest="run_id", metavar="RUN_ID")
    command.add_argument("--retrieval", required=True, help="name, name@N, or a version's ID")
    command.add_argument("--eval-set", required=True, help="name@N, name@latest, or an ID")
    command.add_argument("--split", choices=SPLITS, default="dev")
    command.add_argument(
        "--confirm-test-read",
        action="store_true",
        help="read the test split, and count the read",
    )
    command.add_argument("--seed", type=seed, help="the bootstrap's seed")
    command.add_argument("--resamples", type=_number(int, 100), help="bootstrap resamples")
    command.add_argument(
        "--coverage-threshold",
        type=_number(float, 0.0, above=True),
        help="evidence-coverage-v1's threshold",
    )
    command.add_argument("--shard-size", type=count, help="queries per shard job")
    command.add_argument("--wait", action="store_true", help="wait until it finishes")
    command.add_argument("--rule", choices=RULES, default=DEFAULT_RULE, help="the rule shown")
    for action, handler, help in (
        ("show", eval_show, "show an eval run's headline metrics"),
        ("wait", eval_wait, "wait until an eval run finishes"),
        ("cancel", eval_cancel, "cancel an eval run"),
    ):
        command = leaf(evals, action, handler, help, wait=action == "wait")
        command.add_argument("eval_run_id", metavar="EVAL_RUN_ID")
        command.add_argument("--rule", choices=RULES, default=DEFAULT_RULE, help="the rule shown")
    command = leaf(evals, "search", eval_search, "search eval runs with the search language")
    command.add_argument(
        "filter",
        nargs="?",
        default="",
        help="such as \"metrics.recall@5 > 0.8 and params.rerank.component = 'rerank/none'\"",
    )
    command.add_argument(
        "--order-by", action="append", metavar="KEY", help="such as 'metrics.recall@5 DESC'"
    )
    command.add_argument("--column", action="append", metavar="KEY", help="a column to show")
    command.add_argument("--include-archived", action="store_true")
    command.add_argument("--max-results", type=count)
    command = leaf(evals, "tag", eval_tag, "set an eval run's tags and title")
    command.add_argument("eval_run_id", metavar="EVAL_RUN_ID")
    command.add_argument("pairs", nargs="*", metavar="KEY=VALUE", help="tags to add or replace")
    command.add_argument("--delete", action="append", metavar="KEY", help="a tag to remove")
    command.add_argument("--title", help="its title (empty clears it)")
    command = leaf(evals, "note", eval_note, "set an eval run's Markdown note from a file")
    command.add_argument("eval_run_id", metavar="EVAL_RUN_ID")
    notes = command.add_mutually_exclusive_group(required=True)
    notes.add_argument("--file", metavar="PATH", help="a Markdown file")
    notes.add_argument("--clear", action="store_true", help="remove the note")
    for action, help in (
        ("archive", "hide an eval run from lists and searches (nothing is deleted)"),
        ("restore", "bring an archived eval run back"),
    ):
        command = leaf(evals, action, eval_archive, help)
        command.add_argument("eval_run_id", metavar="EVAL_RUN_ID")
    command = leaf(evals, "compare", eval_compare, "compare two to six eval runs side by side")
    command.add_argument("eval_run_ids", nargs="+", metavar="EVAL_RUN_ID")
    command.add_argument("--baseline", metavar="EVAL_RUN_ID", help="the first by default")
    command.add_argument("--rule", choices=RULES, default=DEFAULT_RULE)
    command.add_argument("--checkpoint", choices=CHECKPOINTS, default=DEFAULT_CHECKPOINT)
    command.add_argument("--metric", action="append", help="such as recall@5, repeatable")
    command.add_argument("--resamples", type=_number(int, 100), help="bootstrap resamples")
    command.add_argument("--seed", type=seed, help="the bootstrap's seed")
    command = leaf(evals, "trace", eval_trace, "show one query's retrieval as a timeline")
    command.add_argument("eval_run_id", metavar="EVAL_RUN_ID")
    command.add_argument("item_id", metavar="ITEM_ID")
    command.add_argument("--rule", choices=RULES, default=DEFAULT_RULE)

    experiments = group("experiment", "plan, start, follow and cancel experiments")
    command = leaf(experiments, "plan", experiment_plan, "check an experiment and show its plan")
    command.add_argument("file", help="YAML or JSON")
    command = leaf(
        experiments, "start", experiment_start, "save an experiment and start it", wait=True
    )
    command.add_argument("file", help="YAML or JSON")
    command.add_argument(
        "--confirm", action="store_true", help="the plan was reviewed: start it (required)"
    )
    command.add_argument("--wait", action="store_true", help="wait until it ends")
    for action, handler, help in (
        ("show", experiment_show, "show an experiment's progress"),
        ("wait", experiment_wait, "wait until an experiment ends"),
        ("cancel", experiment_cancel, "cancel an experiment"),
    ):
        command = leaf(experiments, action, handler, help, wait=action == "wait")
        command.add_argument("experiment_id", metavar="EXPERIMENT_ID")

    profiles = group("profile", "export, download and import knowledge profiles")
    command = leaf(
        profiles, "export", profile_export, "export a trial confirmed on test as a profile"
    )
    command.add_argument("trial_id", metavar="TRIAL_ID")
    command.add_argument("--experiment", required=True, metavar="EXPERIMENT_ID")
    command.add_argument("--name", required=True, help="the profile's name")
    command.add_argument("--description", default="")
    command = leaf(profiles, "download", profile_download, "print a profile version")
    command.add_argument("profile_id", metavar="PROFILE_ID")
    command.add_argument("--output", choices=("yaml", "json"), default="yaml")
    command = leaf(profiles, "import", profile_import, "import a profile file")
    command.add_argument("file", help="YAML or JSON")
    command = leaf(profiles, "alias", profile_alias, "name a profile version (owners)")
    command.add_argument("profile_id", metavar="PROFILE_ID")
    command.add_argument("alias", metavar="ALIAS", help="such as champion")
    command.add_argument("--delete", action="store_true", help="remove the alias instead")

    training = group("train", "mine training sets and train models on them (lab only)")
    command = leaf(
        training, "mine", train_mine, "mine a training set from a dev eval run", wait=True
    )
    command.add_argument("eval_run_id", metavar="EVAL_RUN_ID")
    command.add_argument("--name", help="the training set's name (default: the eval set's)")
    command.add_argument("--max-negatives", type=_number(int, 0), help="per item (default 8)")
    command.add_argument("--max-positives", type=count, help="per item (default 3)")
    command.add_argument(
        "--validation-share", type=_fraction, help="items held out (default 0.2, at most 0.5)"
    )
    command.add_argument("--seed", type=seed, help="the validation split's seed")
    command.add_argument("--wait", action="store_true", help="wait until it's mined")
    command = leaf(training, "export", train_export, "print a training set as JSON Lines")
    command.add_argument("training_set_id", metavar="TRAINING_SET_ID")
    command.add_argument(
        "--as",
        dest="as_",
        choices=("sentence-transformers", "cross-encoder", "chat"),
        default="sentence-transformers",
        help="the rows' shape",
    )
    command.add_argument("--split", choices=("train", "validation", "all"), default="all")
    command.add_argument("--output", metavar="PATH", help="write here instead of stdout")
    command = leaf(training, "start", train_start, "train a model on a training set", wait=True)
    command.add_argument("training_set_id", metavar="TRAINING_SET_ID")
    command.add_argument("--name", required=True, help="the model's name; each run a version")
    command.add_argument(
        "--trainer",
        default="train/query-adapter",
        help="train/query-adapter, train/sentence-transformers or train/cross-encoder",
    )
    command.add_argument(
        "--param", action="append", metavar="KEY=VALUE", help="a trainer parameter, repeatable"
    )
    command.add_argument("--wait", action="store_true", help="wait until it finishes")
    for action, handler, help in (
        ("show", train_show, "show a training run and its epochs"),
        ("wait", train_wait, "wait until a training run finishes"),
        ("cancel", train_cancel, "cancel a training run"),
    ):
        command = leaf(training, action, handler, help, wait=action == "wait")
        command.add_argument("training_run_id", metavar="TRAINING_RUN_ID")

    models = group("model", "list, show, withdraw and download trained models")
    command = leaf(models, "list", model_list, "list trained models, newest first")
    command.add_argument("--kind", choices=("query-adapter", "embedder", "reranker"))
    command.add_argument("--name", help="one model's versions")
    for action, handler, help in (
        ("show", model_show, "show a trained model and how to use it"),
        ("withdraw", model_withdraw, "withdraw a trained model from new runs"),
    ):
        command = leaf(models, action, handler, help)
        command.add_argument("model_id", metavar="MODEL_ID")
    command = leaf(models, "download", model_download, "download a model's zip (files and card)")
    command.add_argument("model_id", metavar="MODEL_ID")
    command.add_argument("--output", required=True, metavar="PATH", help="where to write it")
    command = leaf(models, "compare", model_compare, "compare embedders, rerankers or adapters")
    command.add_argument("--eval-set", required=True, help="name@N, name@latest, or an ID")
    command.add_argument("--role", choices=("embed", "rerank", "adapter"), default="embed")
    command.add_argument("--metric", default="recall@5")
    command.add_argument("--split", choices=SPLITS)
    command = leaf(models, "alias", model_alias, "name a model version with an alias")
    command.add_argument("model_id", metavar="MODEL_ID")
    command.add_argument("alias", metavar="ALIAS", help="such as champion")
    command.add_argument("--delete", action="store_true", help="remove the alias instead")

    tests = group("test", "run test suites, and read their results and reports")
    command = leaf(tests, "run", suite_run, "run a test suite and wait for its result", wait=True)
    command.add_argument("suite", help="a suite file (YAML or JSON), or a saved suite's name[@N]")
    command.add_argument("--junit", metavar="PATH", help="write the JUnit XML report here")
    command.add_argument("--json", metavar="PATH", help="write the stratafi.test-run.v1 here")
    command.add_argument("--markdown", metavar="PATH", help="write the Markdown summary here")
    command.add_argument("--link-base", help="the lab's address, for the Markdown's links")
    command.add_argument("--trigger", choices=("ci", "api"), default="ci", help="who asked")
    command.add_argument("--no-wait", action="store_true", help="queue it and return")
    command = leaf(tests, "validate", suite_validate, "check a suite file without saving it")
    command.add_argument("file", help="YAML or JSON")
    command = leaf(tests, "push", suite_push, "save a suite file as its name's next version")
    command.add_argument("file", help="YAML or JSON")
    for action, handler, help in (
        ("show", suite_show, "show a test run and its checks"),
        ("wait", suite_wait, "wait until a test run ends"),
    ):
        command = leaf(tests, action, handler, help, wait=action == "wait")
        command.add_argument("test_run_id", metavar="TEST_RUN_ID")
    command = leaf(tests, "report", suite_report, "print a finished test run's report")
    command.add_argument("test_run_id", metavar="TEST_RUN_ID")
    command.add_argument("--output", choices=("json", "junit", "markdown"), default="markdown")
    command.add_argument("--link-base", help="the lab's address, for the Markdown's links")
    command = leaf(tests, "list", suite_list, "list test runs, newest first")
    command.add_argument("--suite", help="one suite's runs")

    command = leaf(
        commands,
        "map",
        pipeline_map,
        "print a RAG pipeline's map, with its datastores, as Mermaid (or --format json)",
    )
    command.add_argument("--pipeline", help="a pipeline spec: name, name@N or an ID")
    command.add_argument("--retrieval", help="a retrieval config: name, name@N or an ID")
    command.add_argument("--run", metavar="RUN_ID", help="a pipeline run's map")
    command.add_argument("--eval-run", metavar="EVAL_RUN_ID", help="an eval run's map")

    command = leaf(commands, "compare", compare, "compare two eval runs of one eval set version")
    command.add_argument("a", metavar="EVAL_RUN_A", help="the baseline")
    command.add_argument("b", metavar="EVAL_RUN_B", help="the eval run compared with it")
    command.add_argument("--rule", choices=RULES, default=DEFAULT_RULE)
    command.add_argument("--checkpoint", choices=CHECKPOINTS, default=DEFAULT_CHECKPOINT)
    command.add_argument(
        "--metric",
        action="append",
        help="a metric such as recall@5, repeatable (default: the eval set's headline ones)",
    )
    command.add_argument("--resamples", type=_number(int, 100), help="bootstrap resamples")
    command.add_argument("--seed", type=seed, help="the bootstrap's seed")
    return parser


def main(argv: list[str] | None = None, *, transport: Any = None) -> int:
    """Run a command; its exit code. ``transport`` replaces the HTTP transport (tests)."""
    args = build_parser().parse_args(argv)
    try:
        with Client.from_env(url=args.url, transport=transport) as client:
            return args.handler(client, args)
    except StratafiError as error:
        say(f"stratafi: {error}")
        return error.exit_code
    except KeyboardInterrupt:
        say("stratafi: interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
