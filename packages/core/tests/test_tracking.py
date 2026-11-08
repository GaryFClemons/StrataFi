"""Tracking (unit tier): annotation rules, flattening and the search language."""

import pytest

from stratafi_core.tracking import (
    AnnotationError,
    Clause,
    SearchError,
    TrackedRun,
    available_keys,
    check_note,
    check_title,
    differing_params,
    flatten,
    flatten_metrics,
    flatten_params,
    headline_names,
    matches,
    merge_tags,
    metric_key,
    parse_filter,
    parse_order,
    search,
    sort_runs,
)

# --- annotations --------------------------------------------------------------------------


def test_titles_are_trimmed_cleared_when_empty_and_bounded() -> None:
    assert check_title("  BGE small, no reranker ") == "BGE small, no reranker"
    assert check_title("   ") is None
    assert check_title(None) is None
    with pytest.raises(AnnotationError) as caught:
        check_title("x" * 121)
    assert (caught.value.code, caught.value.field) == ("title_too_long", "title")
    with pytest.raises(AnnotationError, match="title_has_control_characters"):
        check_title("two\nlines")


def test_notes_keep_markdown_and_are_bounded() -> None:
    assert check_note("# Why\n\n- fewer tokens") == "# Why\n\n- fewer tokens"
    assert check_note("") is None
    with pytest.raises(AnnotationError, match="note_too_long"):
        check_note("x" * 20_001)


def test_tags_merge_delete_and_follow_the_rules() -> None:
    merged = merge_tags({"team": "rag", "old": "1"}, set_tags={"owner": "ana"}, delete=["old"])
    assert merged == {"owner": "ana", "team": "rag"}
    assert merge_tags(merged, set_tags={"team": "search"})["team"] == "search"
    for key, code in (
        ("has space", "tag_key_invalid"),
        ("", "tag_key_invalid"),
        ("k" * 65, "tag_key_invalid"),
        ("stratafi.run", "tag_key_reserved"),
        ("Stratafi.Run", "tag_key_reserved"),
    ):
        with pytest.raises(AnnotationError) as caught:
            merge_tags({}, set_tags={key: "v"})
        assert caught.value.code == code
        # The error holds the code and the field, never the key or the value.
        assert key not in str(caught.value) or not key
    with pytest.raises(AnnotationError, match="tag_value_too_long"):
        merge_tags({}, set_tags={"k": "v" * 501})
    with pytest.raises(AnnotationError, match="too_many_tags"):
        merge_tags({f"k{i}": "v" for i in range(50)}, set_tags={"one-more": "v"})
    assert merge_tags({}, set_tags={"a.b/c:d-e_f": "ok"}) == {"a.b/c:d-e_f": "ok"}


# --- flattening ---------------------------------------------------------------------------


def test_flatten_walks_mappings_and_lists() -> None:
    assert flatten({"a": {"b": 1, "c": [{"d": "x"}, 2]}, "e": None, "f": True}) == {
        "a.b": 1,
        "a.c.0.d": "x",
        "a.c.1": 2,
        "e": None,
        "f": True,
    }


STAGES = {
    "parse": {"route": {"default": {"component": "parse/markdown", "params": {}}}},
    "clean": [{"component": "clean/whitespace", "params": {}}],
    "chunk": {"component": "chunk/sections", "params": {"max_tokens": 800}},
    "embed": {"component": "embed/gateway", "params": {"alias": "embed-local"}},
}
RETRIEVAL = {
    "schema": "stratafi.retrieval.v1",
    "name": "hybrid-v1",
    "description": "the service's search",
    "retrievers": {"dense": {"top_n": 50}, "keyword": {"top_n": 50}},
    "fusion": {"method": "rrf", "k": 60},
    "rerank": {"component": "rerank/gateway", "params": {"alias": "rerank-default"}},
    "top_k": 8,
}


def test_params_flatten_stages_config_and_models_without_names() -> None:
    params = flatten_params(
        STAGES, RETRIEVAL, {"embed": {"alias": "embed-local", "revision": "r1"}}
    )
    assert params["chunk.params.max_tokens"] == 800
    assert params["clean.0.component"] == "clean/whitespace"
    assert params["parse.route.default.component"] == "parse/markdown"
    assert params["rerank.component"] == "rerank/gateway"
    assert params["fusion.k"] == 60
    assert params["top_k"] == 8
    assert params["models.embed.revision"] == "r1"
    assert not any(key in params for key in ("schema", "name", "description"))
    assert list(params) == sorted(params)


@pytest.mark.parametrize(
    ("key", "canonical"),
    [
        ("recall@5", "recall@5"),
        ("final.recall@5", "recall@5"),
        ("knowledge-hit-v1.final.recall@5", "recall@5"),
        ("fused.mrr@10", "fused.mrr@10"),
        ("knowledge-hit-v1.dense.hit@all", "dense.hit@all"),
        ("evidence-coverage-v1.final.ndcg@10", "evidence-coverage-v1.final.ndcg@10"),
        ("uplift.knowledge-hit-v1.recall@5", "uplift.recall@5"),
        ("uplift.doc-v1.recall@5", "uplift.doc-v1.recall@5"),
        ("latency.total_ms.p95", "latency.total_ms.p95"),
        ("findability", "findability"),
    ],
)
def test_metric_keys_have_one_shortest_form(key: str, canonical: str) -> None:
    assert metric_key(key) == canonical


def estimate(mean: float) -> dict[str, float]:
    return {"mean": mean, "low": mean - 0.05, "high": mean + 0.05, "n": 60}


SUMMARY = {
    "queries": 90,
    "with_evidence": 74,
    "abstention": 16,
    "k_eval": 5,
    "metrics": {
        "knowledge-hit-v1": {
            "final": {"hit@5": estimate(0.9), "recall@5": estimate(0.85), "mrr@10": estimate(0.7)},
            "fused": {"recall@5": estimate(0.8)},
        },
        "evidence-coverage-v1": {"final": {"recall@5": estimate(0.6)}},
    },
    "uplift": {"knowledge-hit-v1": {"recall@5": estimate(0.05)}},
    "violations": {"avoid": {"count": 2, "of": 10}, "must_not_return": {"count": 0, "of": 90}},
    "where_lost": {"found": 63, "parse": 1, "retrieval": 6, "rerank": 4, "filtered": 0},
    "findability": {"findable": 70, "of": 74, "rate": 0.945946},
    "latency_ms": {"total_ms": {"p50": 12.5, "p95": 30.25, "mean": 14.0}},
}


def test_metrics_flatten_to_canonical_keys() -> None:
    metrics = flatten_metrics(SUMMARY)
    assert metrics["recall@5"] == 0.85
    assert metrics["fused.recall@5"] == 0.8
    assert metrics["evidence-coverage-v1.final.recall@5"] == 0.6
    assert metrics["uplift.recall@5"] == 0.05
    assert metrics["latency.total_ms.p95"] == 30.25
    assert metrics["violations.avoid"] == 2
    assert metrics["violations.must_not_return"] == 0
    assert metrics["lost.retrieval"] == 6
    assert metrics["findability"] == 0.945946
    assert metrics["queries"] == 90
    assert flatten_metrics(None) == {}
    assert headline_names(5) == ["hit@5", "recall@5", "mrr@10", "ndcg@10"]


# --- the search language ------------------------------------------------------------------


def test_a_filter_parses_into_clauses() -> None:
    clauses = parse_filter(
        "metrics.recall@5 > 0.8 AND params.rerank.component = 'rerank/none' "
        'and tags.owner != "ana" and attributes.split IN (\'dev\', "test") '
        "and params.chunk.params.max_tokens <= 512 and tags.note LIKE '%draft%' "
        "and attributes.title ILIKE 'bge%' and tags.x NOT IN ('a') and tags.y NOT LIKE 'b%' "
        "and tags.z IS NULL and tags.w is not null and attributes.test_read = false"
    )
    assert clauses[0] == Clause("metrics", "recall@5", ">", 0.8)
    assert clauses[1] == Clause("params", "rerank.component", "=", "rerank/none")
    assert clauses[2] == Clause("tags", "owner", "!=", "ana")
    assert clauses[3] == Clause("attributes", "split", "in", ("dev", "test"))
    assert clauses[4] == Clause("params", "chunk.params.max_tokens", "<=", 512)
    assert [clause.op for clause in clauses[5:]] == [
        "like", "ilike", "not in", "not like", "is null", "is not null", "="
    ]  # fmt: skip
    assert clauses[-1].value is False
    assert parse_filter("") == []
    assert parse_filter(None) == []


def test_keys_in_backticks_and_singular_entities() -> None:
    (clause,) = parse_filter("param.`parse.route.pdf[scanned].component` = 'parse/docling'")
    assert clause == Clause("params", "parse.route.pdf[scanned].component", "=", "parse/docling")
    (clause,) = parse_filter("metric.dense.hit@all>=0.5")
    assert clause == Clause("metrics", "dense.hit@all", ">=", 0.5)
    (clause,) = parse_filter("tags.k = 'it\\'s'")
    assert clause.value == "it's"


@pytest.mark.parametrize(
    ("text", "code", "position"),
    [
        ("metrics.recall@5 > 'x", "unterminated_string", 19),
        ("recall@5 > 0.8", "unknown_entity", 0),
        ("metrics.recall@5", "expected_operator", 16),
        ("metrics.recall@5 >", "expected_value", 18),
        ("metrics.recall@5 > 0.8 metrics.mrr@10 > 0", "expected_and", 23),
        ("metrics.recall@5 > 0.8 or tags.a = 'b'", "or_unsupported", 23),
        ("tags.a LIKE 5", "expected_string", 12),
        ("tags.a IN 'b'", "expected_list", 10),
        ("tags.a IN ('b'", "expected_list", 14),
        ("tags.a IS 'b'", "expected_null", 10),
        ("tags.a ~ 'b'", "unexpected_character", 7),
        ("and tags.a = 'b'", "expected_key", 0),
        ("tags.a NOT = 'b'", "expected_operator", 11),
    ],
)
def test_errors_say_where_never_what(text: str, code: str, position: int) -> None:
    with pytest.raises(SearchError) as caught:
        parse_filter(text)
    assert (caught.value.code, caught.value.position) == (code, position)
    assert "'b'" not in str(caught.value)


def test_long_filters_and_many_clauses_are_refused() -> None:
    with pytest.raises(SearchError, match="filter_too_long"):
        parse_filter("tags.a = 'b' and " * 300)
    with pytest.raises(SearchError, match="too_many_clauses"):
        parse_filter(" and ".join(["tags.a = 'b'"] * 51))


def run(identifier: str, created: str, **values: dict) -> TrackedRun:
    return TrackedRun(
        id=identifier,
        attributes={"created_at": created, **values.get("attributes", {})},
        params=values.get("params", {}),
        metrics=values.get("metrics", {}),
        tags=values.get("tags", {}),
    )


RUNS = [
    run(
        "a",
        "2026-11-05T10:00:00+00:00",
        attributes={"status": "succeeded", "split": "dev", "title": "BGE small"},
        params={"rerank.component": "rerank/gateway", "fusion.k": 60, "top_k": 8},
        metrics={"recall@5": 0.85, "latency.total_ms.p95": 30.0},
        tags={"owner": "ana"},
    ),
    run(
        "b",
        "2026-11-05T11:00:00+00:00",
        attributes={"status": "succeeded", "split": "dev", "title": "no reranker"},
        params={"rerank.component": "rerank/none", "fusion.k": 60, "top_k": 5},
        metrics={"recall@5": 0.78, "latency.total_ms.p95": 9.0},
        tags={"owner": "ben", "draft": "yes"},
    ),
    run(
        "c",
        "2026-11-05T12:00:00+00:00",
        attributes={"status": "failed", "split": "test", "title": None},
        params={"rerank.component": "rerank/gateway", "fusion.k": "40"},
        metrics={},
        tags={},
    ),
]


def ids(text: str, order: list[str] | None = None) -> list[str]:
    return [found.id for found in search(RUNS, filter=text, order_by=order or [])]


def test_filters_evaluate_as_sql_would() -> None:
    assert ids("metrics.recall@5 > 0.8") == ["a"]
    # The default rule and checkpoint, written out, are the same metric.
    assert ids("metrics.knowledge-hit-v1.final.recall@5 > 0.8") == ["a"]
    assert ids("params.rerank.component = 'rerank/gateway'") == ["c", "a"]
    # Numbers compare as numbers, quoted or not, whichever side holds the text.
    assert ids("params.fusion.k = 60") == ["b", "a"]
    assert ids("params.fusion.k < '50'") == ["c"]
    assert ids("params.top_k IN (5, 6)") == ["b"]
    assert ids("params.top_k NOT IN (5)") == ["a"]  # c has no top_k: a missing key never matches
    assert ids("tags.owner != 'ana'") == ["b"]
    assert ids("tags.draft IS NULL") == ["c", "a"]
    assert ids("tags.draft IS NOT NULL") == ["b"]
    assert ids("attributes.title LIKE '%rerank%'") == ["b"]
    assert ids("attributes.title ILIKE 'bge%'") == ["a"]
    assert ids("attributes.title LIKE 'bge%'") == []
    assert ids("attributes.title NOT LIKE 'BGE%'") == ["b"]
    assert ids("attributes.created_at > '2026-11-05T10:30'") == ["c", "b"]
    assert ids("attributes.status = 'succeeded' and attributes.split = 'dev'") == ["b", "a"]
    assert ids("metrics.unknown@5 > 0") == []


def test_ordering_puts_missing_values_last_either_way() -> None:
    assert ids("", ["metrics.recall@5 DESC"]) == ["a", "b", "c"]
    assert ids("", ["metrics.recall@5"]) == ["b", "a", "c"]
    assert ids("", ["metrics.latency.total_ms.p95 ASC"]) == ["b", "a", "c"]
    assert ids("", ["params.rerank.component", "metrics.recall@5 DESC"]) == ["a", "c", "b"]
    # Ties fall back to the newest first.
    assert ids("") == ["c", "b", "a"]
    with pytest.raises(SearchError, match="expected_direction"):
        parse_order(["metrics.recall@5 DOWN"])
    with pytest.raises(SearchError, match="unknown_entity"):
        parse_order(["recall@5 DESC"])
    with pytest.raises(SearchError, match="too_many_orders"):
        parse_order(["tags.a"] * 6)


def test_matches_and_sort_take_parsed_inputs() -> None:
    clauses = parse_filter("tags.owner = 'ana'")
    assert [found.id for found in RUNS if matches(found, clauses)] == ["a"]
    assert [found.id for found in sort_runs(RUNS, parse_order(["attributes.title"]))] == [
        "a", "b", "c"
    ]  # fmt: skip


def test_keys_and_differing_params_for_pickers() -> None:
    keys = available_keys(RUNS)
    assert keys["metrics"] == ["latency.total_ms.p95", "recall@5"]
    assert keys["tags"] == ["draft", "owner"]
    assert "fusion.k" in keys["params"]
    # fusion.k is 60 twice and "40" once; top_k is missing from c.
    assert differing_params(RUNS) == ["fusion.k", "rerank.component", "top_k"]
    assert differing_params(RUNS[:1]) == []
    assert RUNS[0].value("metrics.final.recall@5") == 0.85
    assert RUNS[0].value("tags.missing") is None
