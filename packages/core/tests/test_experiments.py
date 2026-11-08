"""Experiments: the spec, factor paths, expansion with exclusions and
strategies, trial resolution, the unique work a grid needs, estimates, successive halving's
plan, and a trial's measures."""

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from stratafi_core.experiments import (
    History,
    PlanEntry,
    budget_check,
    check_constraints,
    dump_experiment,
    estimate,
    halving_plan,
    level_label,
    matches,
    measures,
    parse_experiment,
    parse_path,
    parse_ref,
    plan_trials,
    read_path,
    set_path,
    unique_work,
    variant_description,
    variant_name,
)
from stratafi_core.hashing import content_hash
from stratafi_core.planner import EntryInput
from stratafi_core.specs import SpecInvalid, resolve_pipeline, resolve_retrieval
from stratafi_testing.components import FixedParams, stub, stub_registry

REGISTRY = stub_registry()
for _component in (
    stub("parse/stub-a", "parse"),
    stub("parse/stub-b", "parse"),
    stub("chunk/stub-c", "chunk", FixedParams),
    stub("clean/stub-corpus", "clean", corpus_scope=True),
):
    REGISTRY.register(_component)

# Every document takes the default parse route, so a factor on it changes every parse.
BASE_PIPELINE = resolve_pipeline(
    {
        "schema": "stratafi.pipeline.v1",
        "name": "base",
        "stages": {
            "scan": {"checks": ["scan/content-safety", "scan/clamav"]},
            "parse": {"route": {"default": "parse/markdown"}},
            "embed": {"component": "embed/gateway", "params": {"alias": "embed-a"}},
        },
    },
    REGISTRY,
)
BASE_RETRIEVAL = resolve_retrieval(
    {"schema": "stratafi.retrieval.v1", "name": "hybrid", "rerank": "rerank/none"}, REGISTRY
)


def experiment(**changes: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema": "stratafi.experiment.v1",
        "name": "grid",
        "corpus": "halcyon@1",
        "eval_set": {"name": "halcyon-retrieval", "version": 1},
        "base": {"pipeline": "base", "retrieval": "hybrid"},
        "objective": {"metric": "recall@5"},
        "factors": {"chunk.params.max_tokens": [400, 800]},
    }
    document.update(changes)
    return document


def plan(**changes: Any):
    return plan_trials(
        parse_experiment(experiment(**changes)), BASE_PIPELINE, BASE_RETRIEVAL, REGISTRY
    )


GRID = {
    "parse.route.default": ["parse/markdown", "parse/stub-a", "parse/stub-b"],
    "chunk": [
        {"component": "chunk/sections", "params": {"max_tokens": 800}},
        {"component": "chunk/fixed-tokens", "params": {"size": 512, "overlap": 64}},
        {"component": "chunk/stub-c", "params": {"size": 256}},
    ],
    "embed.params.alias": ["embed-a", "embed-b"],
    "rerank": ["rerank/none", {"component": "rerank/gateway", "params": {"alias": "r1"}}],
}


def entries(count: int, *, media_type: str = "text/markdown") -> list[PlanEntry]:
    return [
        PlanEntry(
            EntryInput(f"docs/d{n}.md", f"doc-{n}", content_hash(f"d{n}"), ".md"),
            media_type=media_type,
            size_bytes=4000,
        )
        for n in range(count)
    ]


def nothing_known(keys: Sequence[str]) -> Mapping[str, Mapping[str, Any]]:
    return {}


# --- the spec and its paths -------------------------------------------------------------------


def test_references_name_a_version_or_the_latest() -> None:
    assert parse_ref("halcyon") == ("halcyon", None)
    assert parse_ref("halcyon@3") == ("halcyon", 3)
    assert parse_ref("halcyon@latest") == ("halcyon", None)
    for bad in ("Halcyon", "halcyon@0", "halcyon@v2", "halcyon@"):
        with pytest.raises(ValueError, match="reference"):
            parse_ref(bad)


def test_paths_go_into_the_pipeline_or_the_retrieval_config() -> None:
    route = parse_path('parse.route."application/pdf"')
    assert route.target == "pipeline"
    assert route.keys == ("stages", "parse", "route", "application/pdf")
    assert route.stage == "parse"
    assert parse_path("embed.params.alias").keys == ("stages", "embed", "params", "alias")
    assert parse_path("stages.chunk.params.size").keys == ("stages", "chunk", "params", "size")
    assert parse_path("pipeline.stages.clean.0").keys == ("stages", "clean", "0")
    rerank = parse_path("rerank")
    assert (rerank.target, rerank.keys, rerank.stage) == ("retrieval", ("rerank",), "query")
    assert parse_path("retrieval.fusion.k").keys == ("fusion", "k")
    for bad, code in [
        ("name", "unknown_path"),
        ("stages.nothing", "unknown_path"),
        ("chunk..size", "invalid_path"),
        ('parse.route."unclosed', "invalid_path"),
        ("", "invalid_path"),
    ]:
        with pytest.raises(ValueError, match=code):
            parse_path(bad)


def test_levels_replace_a_reference_and_keep_its_other_keys() -> None:
    chunk = {"component": "c/a", "params": {"x": 1}, "index_text": "{text}"}
    document = {"stages": {"chunk": chunk}}
    set_path(document, ("stages", "chunk"), "chunk/fixed-tokens")
    assert document["stages"]["chunk"] == {
        "component": "chunk/fixed-tokens",
        "params": {},
        "index_text": "{text}",
    }
    set_path(document, ("stages", "chunk"), {"params": {"size": 9}})
    assert document["stages"]["chunk"]["component"] == "chunk/fixed-tokens"
    assert document["stages"]["chunk"]["params"] == {"size": 9}
    set_path(document, ("stages", "parse", "route", "application/pdf"), "parse/stub-a")
    assert document["stages"]["parse"]["route"]["application/pdf"] == "parse/stub-a"
    assert read_path(document, ("stages", "chunk", "params", "size")) == 9
    assert read_path(document, ("stages", "missing", "x")) is None
    with pytest.raises(ValueError, match="no_such_path"):
        set_path({"clean": [{"component": "clean/whitespace"}]}, ("clean", "3"), "clean/unicode")


def test_exclusions_match_components_mappings_and_numbers() -> None:
    reference = {"component": "chunk/fixed-tokens", "params": {"size": 512, "overlap": 64}}
    assert matches(reference, "chunk/fixed-tokens")
    assert not matches(reference, "chunk/stub-c")
    assert matches(reference, {"params": {"size": 512}})
    assert not matches(reference, {"params": {"size": 256}})
    assert matches(0.1, 0.10)
    assert not matches(1, True)


def test_levels_have_readable_labels() -> None:
    assert level_label("embed-local") == "embed-local"
    assert level_label(0.1) == "0.1"
    assert level_label(None) == "null"
    reference = {"component": "chunk/fixed-tokens", "params": {"size": 512, "overlap": 64}}
    assert level_label(reference) == "chunk/fixed-tokens overlap=64 size=512"
    assert len(level_label({"component": "a/b", "params": {"x": "y" * 200}})) == 80


def test_a_spec_with_a_bad_shape_names_paths_and_codes() -> None:
    with pytest.raises(SpecInvalid) as raised:
        parse_experiment(experiment(objective={"metric": "accuracy"}, budget={"max_trials": 0}))
    assert {issue.path for issue in raised.value.issues} >= {
        "objective.metric",
        "budget.max_trials",
    }
    with pytest.raises(SpecInvalid):
        parse_experiment(experiment(strategy={"method": "random"}))
    with pytest.raises(SpecInvalid):
        parse_experiment(experiment(constraints=[{"metric": "latency_ms.p95"}]))
    spec = parse_experiment(experiment(strategy="grid"))
    assert spec.search.method == "grid"
    assert spec.eval_set.split == "dev"


# --- expanding and resolving -----------------------------------------------------------------


def test_a_grid_expands_in_factor_order_and_resolves_every_trial() -> None:
    planned = plan(factors=GRID)
    assert planned.issues == []
    assert planned.grid == 36
    assert len(planned.trials) == 36
    first, last = planned.trials[0], planned.trials[-1]
    assert first.levels == (0, 0, 0, 0)
    assert last.levels == (2, 2, 1, 1)
    assert planned.trials[1].levels == (0, 0, 0, 1)  # the last factor varies fastest
    assert first.values(planned.factors) == {
        "parse.route.default": "parse/markdown",
        "chunk": "chunk/sections max_tokens=800",
        "embed.params.alias": "embed-a",
        "rerank": "rerank/none",
    }
    assert last.pipeline.stages.parse.route["default"].component == "parse/stub-b"
    assert last.pipeline.stages.chunk.component == "chunk/stub-c"
    # The chunk stage's index text stays the base's.
    assert last.pipeline.stages.chunk.index_text == BASE_PIPELINE.stages.chunk.index_text
    assert last.retrieval.rerank.params == {"alias": "r1"}
    assert len(planned.pipelines) == 18  # the reranker is query time: two trials share each
    assert len({trial.pipeline_hash for trial in planned.trials}) == 18


def test_exclusions_drop_combinations_and_random_search_draws_some() -> None:
    planned = plan(
        factors=GRID,
        exclude=[{"chunk.component": "chunk/stub-c", "parse.route.default": "parse/stub-a"}],
    )
    assert planned.excluded == 4
    assert len(planned.trials) == 32
    assert not any(
        trial.pipeline.stages.chunk.component == "chunk/stub-c"
        and trial.pipeline.stages.parse.route["default"].component == "parse/stub-a"
        for trial in planned.trials
    )
    drawn = plan(factors=GRID, strategy={"method": "random", "n": 7, "seed": 3})
    again = plan(factors=GRID, strategy={"method": "random", "n": 7, "seed": 3})
    assert len(drawn.trials) == 7
    assert [t.grid_index for t in drawn.trials] == [t.grid_index for t in again.trials]
    assert [t.grid_index for t in drawn.trials] == sorted(t.grid_index for t in drawn.trials)
    assert [t.ordinal for t in drawn.trials] == list(range(7))


def test_bad_factors_are_issues_not_errors() -> None:
    planned = plan(
        factors={
            "chunk.params.max_tokens": [400, 400],
            "chunk": ["chunk/fixed-tokens"],
            "nowhere.x": [1],
        }
    )
    codes = {(issue.path, issue.code) for issue in planned.issues}
    assert ("factors.chunk.params.max_tokens.1", "duplicate_level") in codes
    assert ("factors.chunk", "overlapping_factor") in codes
    assert ("factors.*", "unknown_path") in codes
    assert planned.trials == []


def test_a_trial_that_does_not_validate_is_an_issue() -> None:
    planned = plan(factors={"chunk.params.max_tokens": [800, -1]})
    assert [trial.levels for trial in planned.trials] == [(0,)]
    assert any(
        issue.path == "trials.1.pipeline.stages.chunk.params.max_tokens" for issue in planned.issues
    )
    duplicate = plan(factors={"chunk.params.max_tokens": [800, 800.0]})
    assert any(issue.code == "duplicate_trial" for issue in duplicate.issues)


def test_excluding_everything_is_an_issue() -> None:
    planned = plan(exclude=[{"chunk.component": "chunk/sections"}])
    assert [issue.code for issue in planned.issues] == ["every_trial_excluded"]


def test_variants_are_named_by_their_base_and_configuration() -> None:
    name = variant_name("baseline-v1", "ab" * 32)
    assert name == "baseline-v1.xabababab" + "ab"
    assert len(variant_name("a" * 64, "c" * 64)) == 64
    before = {"name": "b", "stages": {"chunk": {"params": {"max_tokens": 800}}}}
    after = {"name": "b.x1", "stages": {"chunk": {"params": {"max_tokens": 400}}}}
    assert variant_description("b", 2, before, after) == (
        "An experiment's variant of b v2: stages.chunk.params.max_tokens"
    )


# --- the unique work --------------------------------------------------------------------------


def test_the_unique_work_of_a_3x3x2x2_grid_matches_the_theory() -> None:
    planned = plan(factors=GRID)
    documents = 5
    work = unique_work(
        planned.trials, REGISTRY, entries(documents), model_envs={}, lookup=nothing_known
    )
    stages = {stage: item.as_dict() for stage, item in work.stages.items()}
    # Every trial alone: 36 trials, each 7 document tasks per document and two indexes.
    assert {stage: counts["tasks"] for stage, counts in stages.items()} == {
        "extract": 36 * documents,
        "scan": 36 * documents,
        "parse": 36 * documents,
        "clean": 36 * documents,
        "chunk": 36 * documents,
        "enrich": 36 * documents,
        "embed": 36 * documents,
        "index": 36 * 2,
    }
    # Shared prefixes run once: 3 parsers, 3 x 3 chunkings, 3 x 3 x 2 embeddings; the keyword
    # index reads chunks only (9), the dense one chunks and vectors (18). Rerankers re-embed
    # nothing.
    assert {stage: counts["unique"] for stage, counts in stages.items()} == {
        "extract": documents,
        "scan": documents,
        "parse": 3 * documents,
        "clean": 3 * documents,
        "chunk": 9 * documents,
        "enrich": 9 * documents,
        "embed": 18 * documents,
        "index": 18 + 9,
    }
    assert all(counts["cached"] == 0 for counts in stages.values())
    assert len(work.pipelines) == 18


def corpus_work(ranks: Sequence[tuple[str, ...]]) -> Any:
    """Two parsers' unique work over three documents, with a corpus-wide clean rule."""
    clean = [ref.model_dump(mode="json") for ref in BASE_PIPELINE.stages.clean]
    base = resolve_pipeline(
        {
            "schema": "stratafi.pipeline.v1",
            "name": "base",
            "stages": {
                "scan": {"checks": ["scan/content-safety", "scan/clamav"]},
                "parse": {"route": {"default": "parse/markdown"}},
                "clean": [*clean, "clean/stub-corpus"],
                "embed": {"component": "embed/gateway", "params": {"alias": "embed-a"}},
            },
        },
        REGISTRY,
    )
    parsers = {"parse.route.default": ["parse/stub-a", "parse/stub-b"]}
    spec = parse_experiment(experiment(factors=parsers))
    planned = plan_trials(spec, base, BASE_RETRIEVAL, REGISTRY)
    corpus = [
        PlanEntry(item.entry, media_type=item.media_type, size_bytes=4000, rank=rank)
        for item, rank in zip(entries(len(ranks)), ranks, strict=True)
    ]
    return unique_work(planned.trials, REGISTRY, corpus, model_envs={}, lookup=nothing_known)


def test_a_corpus_wide_clean_rule_waits_for_every_parse_as_a_run_does() -> None:
    """A clean rule that compares documents (near duplicates) makes each entry wait at
    Clean, as in a run; its clean task then reads every parse key and rank, so each parser's
    corpus gets its own clean tasks, and the ranks are part of their keys."""
    ranks = [("2026-01-01", "1", "d0"), ("2026-02-01", "2", "d1"), ("", "", "d2")]
    work = corpus_work(ranks)
    unique = {stage: item.unique for stage, item in work.stages.items()}
    assert (unique["parse"], unique["clean"], unique["embed"]) == (6, 6, 6)
    assert work.as_dict()["stopped"] == {}
    assert set(corpus_work(ranks[::-1]).tasks) != set(work.tasks)


def test_planning_reuses_what_exists_and_what_scans_decided() -> None:
    planned = plan(factors={"chunk.params.max_tokens": [400, 800]})
    corpus = entries(3)
    first = unique_work(planned.trials, REGISTRY, corpus, model_envs={}, lookup=nothing_known)
    made: dict[str, dict[str, Any]] = {}
    summaries = {
        "extract": {"media_type": "text/markdown", "traits": {}},
        "scan": {"decision": "continue"},
        "parse": {"chars": 100},
        "clean": {"chars": 100},
    }
    for key, ref in first.tasks.items():
        if ref.stage in summaries:
            made[key] = summaries[ref.stage]
    # One document's scan excluded it.
    excluded = next(
        key for key, ref in first.tasks.items() if ref.stage == "scan" and ref.entry == "docs/d0.md"
    )
    made[excluded] = {"decision": "exclude"}

    def lookup(keys: Sequence[str]) -> Mapping[str, Mapping[str, Any]]:
        return {key: made[key] for key in keys if key in made}

    second = unique_work(planned.trials, REGISTRY, corpus, model_envs={}, lookup=lookup)
    stages = {stage: item.as_dict() for stage, item in second.stages.items()}
    assert stages["parse"] == {"tasks": 4, "unique": 2, "cached": 2, "to_run": 0}
    assert stages["chunk"] == {"tasks": 4, "unique": 4, "cached": 0, "to_run": 4}
    assert second.as_dict()["stopped"] == {"excluded": 2}


def test_a_model_revision_is_part_of_the_work() -> None:
    planned = plan(factors={"chunk.params.max_tokens": [400]})
    corpus = entries(2)
    pipeline_hash = planned.trials[0].pipeline_hash
    one = unique_work(
        planned.trials,
        REGISTRY,
        corpus,
        model_envs={pipeline_hash: {"embed": {"revision": "r1"}}},
        lookup=nothing_known,
    )
    two = unique_work(
        planned.trials,
        REGISTRY,
        corpus,
        model_envs={pipeline_hash: {"embed": {"revision": "r2"}}},
        lookup=nothing_known,
    )
    embeds = {key for key, ref in one.tasks.items() if ref.stage == "embed"}
    assert embeds.isdisjoint(key for key, ref in two.tasks.items() if ref.stage == "embed")
    chunks = {key for key, ref in one.tasks.items() if ref.stage == "chunk"}
    assert chunks == {key for key, ref in two.tasks.items() if ref.stage == "chunk"}


def test_routing_by_type_shares_work_a_factor_does_not_touch() -> None:
    planned = plan(factors={'parse.route."application/pdf"': ["parse/stub-a", "parse/stub-b"]})
    corpus = entries(2) + entries(1, media_type="application/pdf")
    work = unique_work(planned.trials, REGISTRY, corpus, model_envs={}, lookup=nothing_known)
    # The Markdown documents take the default route either way; only the PDF parses twice.
    assert work.stages["parse"].unique == 2 + 2


# --- estimates and measures ------------------------------------------------------------------


def test_estimates_use_history_then_defaults_and_the_price_table() -> None:
    planned = plan(factors=GRID)
    corpus = entries(2)
    work = unique_work(planned.trials, REGISTRY, corpus, model_envs={}, lookup=nothing_known)
    history = History(task_ms={("parse/stub-a", "1.0.0"): 1000.0}, query_ms=10.0)
    estimated = estimate(
        work,
        planned.trials,
        REGISTRY,
        corpus,
        queries=90,
        shard_size=25,
        history=history,
        prices={"model_gateway": 0.02},
        slots={"cpu": 2, "io": 4},
    )
    assert estimated["basis"]["parse"] == "default"  # two of three parsers have no history
    assert estimated["basis"]["eval"] == "history"
    assert estimated["eval"] == {
        "trials": 36,
        "to_run": 36,
        "queries": 90,
        "shards": 4 * 36,
        "query_share": 1.0,
    }
    # The stubs declare no egress, so nothing here goes to the gateway's price.
    assert estimated["cost_usd"]["ingest"] == 0
    assert estimated["cpu_seconds"] > 36 * 90 * 0.010
    assert estimated["wall_seconds"] < estimated["cpu_seconds"]
    some = estimate(
        work,
        planned.trials,
        REGISTRY,
        corpus,
        queries=90,
        shard_size=25,
        history=history,
        prices={},
        slots={},
        evaluated={trial.ordinal for trial in planned.trials[:30]},
    )
    assert some["eval"]["to_run"] == 6
    spec = parse_experiment(experiment(budget={"max_trials": 10, "max_cost_usd": 0}))
    check = budget_check(spec, 36, estimated)
    assert check["trials"] == {"planned": 36, "max": 10, "over": True}
    assert check["cost_usd"]["over"] is False


def test_successive_halving_plans_its_rungs_and_scores_a_share_of_the_queries() -> None:
    """Halving keeps every combination (or draws ``n``), lays out its rungs for the
    trials the budget admits, and the estimate scores its nominal share of the queries."""
    halving = {"method": "successive-halving", "eta": 3, "min_queries": 10, "seed": 5}
    spec = parse_experiment(experiment(factors=GRID, strategy=halving))
    planned = plan_trials(spec, BASE_PIPELINE, BASE_RETRIEVAL, REGISTRY)
    assert len(planned.trials) == 36
    laid_out = halving_plan(spec, 36, 90)
    assert laid_out == {
        "eta": 3,
        "min_queries": 10,
        "seed": 5,
        "rungs": [
            {"queries": 10, "keep": 12},
            {"queries": 30, "keep": 4},
            {"queries": 90, "keep": 4},
        ],
        "cost": 36 * 10 + 12 * 20 + 4 * 60,
        "evaluations": 36 * 10 + 12 * 30 + 4 * 90,
        "grid": 36 * 90,
        "share": 0.2593,
    }
    budgeted = parse_experiment(
        experiment(factors=GRID, strategy="successive-halving", budget={"max_trials": 9})
    )
    assert (halving_plan(budgeted, 36, 90) or {})["rungs"][0] == {"queries": 10, "keep": 3}
    assert halving_plan(parse_experiment(experiment()), 2, 90) is None
    drawn = plan(factors=GRID, strategy={**halving, "n": 12})
    assert len(drawn.trials) == 12
    corpus = entries(2)
    work = unique_work(planned.trials, REGISTRY, corpus, model_envs={}, lookup=nothing_known)
    arguments: dict[str, Any] = {
        "queries": 90,
        "shard_size": 25,
        "history": History(query_ms=10.0),
        "prices": {},
        "slots": {},
    }
    full = estimate(work, planned.trials, REGISTRY, corpus, **arguments)
    share = estimate(work, planned.trials, REGISTRY, corpus, **arguments, query_share=0.2593)
    assert share["eval"]["query_share"] == 0.2593
    assert share["eval"]["shards"] < full["eval"]["shards"]
    assert share["cpu_seconds"] < full["cpu_seconds"]
    # A stored document keeps every default, and reads back as it was.
    for written in (spec, parse_experiment(experiment())):
        assert parse_experiment(dump_experiment(written)).search == written.search
    # Halving's options are its own, and the strategy decides which queries are scored.
    with pytest.raises(SpecInvalid):
        parse_experiment(experiment(strategy={"method": "grid", "eta": 2}))
    with pytest.raises(SpecInvalid):
        parse_experiment(experiment(eval_options={"sample": {"size": 10}}))


def test_measures_come_from_an_eval_runs_metrics() -> None:
    metrics = {
        "metrics": {
            "knowledge-hit-v1": {
                "final": {"recall@5": {"mean": 0.8, "low": 0.7, "high": 0.9, "n": 80}},
            }
        },
        "latency_ms": {"total_ms": {"p50": 12.0, "p95": 40.0}},
        "violations": {"must_not_return": {"count": 1, "of": 90}, "avoid": {"count": 0, "of": 5}},
        "cost": {"per_1k_queries": 0.3},
        "findability": {"rate": 0.97},
    }
    spec = parse_experiment(
        experiment(
            constraints=[
                {"metric": "latency_ms.p95", "max": 50},
                {"metric": "must_not_return_violations", "max": 0},
                {"metric": "recall@5", "min": 0.75},
                {"metric": "ingest_cost_usd", "max": 1},
            ]
        )
    )
    found = measures(metrics, objective=spec.objective, ingest_cost=None)
    assert found["recall@5"] == 0.8
    assert found["latency_ms.p95"] == 40.0
    assert found["cost_per_1k_queries_usd"] == 0.3
    results = check_constraints(spec.constraints, found)
    assert [result["passed"] for result in results] == [True, False, True, False]
