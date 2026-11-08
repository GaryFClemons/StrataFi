"""Pipeline specs and retrieval configs: validation with precise paths, default
filling, hashes and diffs. The registry is ``stratafi_testing.components``' stubs."""

import pytest

from stratafi_core.components import ComponentRegistry
from stratafi_core.specs import (
    SpecInvalid,
    accepts,
    component_refs,
    diff_documents,
    document_hash,
    dump_document,
    resolve_pipeline,
    resolve_retrieval,
    route_for,
    spec_hash,
)
from stratafi_testing.components import stub_registry

REGISTRY = stub_registry()


def pipeline(**stages: object) -> dict:
    return {"schema": "stratafi.pipeline.v1", "name": "test", "stages": stages}


def issues(data: dict, registry: ComponentRegistry = REGISTRY) -> list[tuple[str, str]]:
    with pytest.raises(SpecInvalid) as raised:
        resolve_pipeline(data, registry)
    return [(issue.path, issue.code) for issue in raised.value.issues]


# --- defaults and hashes ----------------------------------------------------------------------


def test_a_minimal_spec_resolves_to_the_baseline_with_every_default() -> None:
    resolved = dump_document(resolve_pipeline(pipeline(), REGISTRY))
    stages = resolved["stages"]
    assert stages["chunk"] == {
        "component": "chunk/sections",
        "params": {
            "max_tokens": 400,
            "min_tokens": 200,
            "overlap": 0,
            "token_estimator": "chars/4",
        },
        "index_text": "{title}\n{heading_path}\n{text}",
    }
    assert stages["parse"]["route"] == {
        "default": {"component": "parse/markitdown", "params": {}},
        "docx": {"component": "parse/markitdown", "params": {}},
        "html": {"component": "parse/markitdown", "params": {}},
        "md": {"component": "parse/markdown", "params": {}},
        "pdf": {"component": "parse/pypdf", "params": {}},
        "pptx": {"component": "parse/markitdown", "params": {}},
        "txt": {"component": "parse/markdown", "params": {}},
        "xlsx": {"component": "parse/markitdown", "params": {}},
    }
    assert [check["component"] for check in stages["scan"]["checks"]] == [
        "scan/content-safety",
        "scan/dedupe",
    ]
    assert set(stages["scan"]["on_outcome"].values()) == {"exclude"}
    assert stages["embed"] == {
        "component": "embed/gateway",
        "params": {"alias": "embed-default", "batch_size": 32},
    }


def test_hashes_are_stable_across_key_order_and_default_filling() -> None:
    terse = pipeline(chunk={"params": {"max_tokens": 400}}, clean=["clean/whitespace"])
    explicit = {
        "stages": {
            "clean": [{"params": {}, "component": "clean/whitespace"}],
            "chunk": {
                "index_text": "{title}\n{heading_path}\n{text}",
                "params": {
                    "token_estimator": "chars/4",
                    "overlap": 0,
                    "min_tokens": 200,
                    "max_tokens": 400,
                },
                "component": "chunk/sections",
            },
            "scan": {"on_outcome": {"infected": "exclude"}},
        },
        "name": "test",
        "schema": "stratafi.pipeline.v1",
    }
    a, b = resolve_pipeline(terse, REGISTRY), resolve_pipeline(explicit, REGISTRY)
    assert spec_hash(a) == spec_hash(b)
    assert document_hash(a) == document_hash(b)
    changed = resolve_pipeline(pipeline(chunk={"params": {"max_tokens": 300}}), REGISTRY)
    assert spec_hash(changed) != spec_hash(a)


def test_the_spec_hash_ignores_the_name_and_description_but_the_document_hash_doesnt() -> None:
    a = resolve_pipeline({**pipeline(), "name": "one"}, REGISTRY)
    b = resolve_pipeline({**pipeline(), "name": "two", "description": "Same config"}, REGISTRY)
    assert spec_hash(a) == spec_hash(b)
    assert document_hash(a) != document_hash(b)


def test_resolving_a_resolved_spec_changes_nothing() -> None:
    once = resolve_pipeline(pipeline(chunk={"params": {"overlap": 32}}), REGISTRY)
    twice = resolve_pipeline(dump_document(once), REGISTRY)
    assert dump_document(once) == dump_document(twice)


def test_bare_component_ids_mean_default_parameters() -> None:
    resolved = resolve_pipeline(pipeline(embed="embed/gateway", enrich="enrich/none"), REGISTRY)
    assert resolved.stages.embed.params == {"alias": "embed-default", "batch_size": 32}


def test_an_incomplete_scan_policy_excludes_what_it_doesnt_name() -> None:
    resolved = resolve_pipeline(pipeline(scan={"on_outcome": {"duplicate": "flag"}}), REGISTRY)
    assert resolved.stages.scan.on_outcome["duplicate"] == "flag"
    assert resolved.stages.scan.on_outcome["pii"] == "exclude"


# --- precise paths ----------------------------------------------------------------------------


def test_bad_parameters_name_their_path() -> None:
    found = issues(pipeline(chunk={"params": {"max_tokens": 0, "overlap": -1}}))
    assert ("stages.chunk.params.max_tokens", "greater_than_equal") in found
    assert ("stages.chunk.params.overlap", "greater_than_equal") in found


def test_unknown_parameters_name_their_path() -> None:
    assert issues(pipeline(embed={"component": "embed/gateway", "params": {"modle": "x"}})) == [
        ("stages.embed.params.modle", "extra_forbidden")
    ]


def test_unknown_parameter_names_that_arent_identifiers_are_masked() -> None:
    found = issues(pipeline(embed={"component": "embed/gateway", "params": {"Free text!": 1}}))
    assert found == [("stages.embed.params.*", "extra_forbidden")]


def test_unknown_components_and_wrong_stages_name_their_path() -> None:
    found = issues(
        pipeline(
            clean=["clean/whitespace", "clean/nonexistent"],
            enrich={"component": "chunk/fixed-tokens"},
        )
    )
    assert ("stages.clean.1.component", "unknown_component") in found
    assert ("stages.enrich.component", "wrong_stage") in found


def test_malformed_component_references_name_their_path() -> None:
    assert issues(pipeline(clean=[{"component": "Not An Id"}])) == [
        ("stages.clean.0.component", "string_pattern_mismatch")
    ]
    assert issues(pipeline(clean=[42])) == [("stages.clean.0", "invalid_type")]


def test_the_parse_route_needs_a_default_and_parsers_that_accept_their_types() -> None:
    found = issues(
        pipeline(
            parse={
                "route": {
                    "application/pdf": "parse/pypdf",
                    "pdf[scanned]": "parse/stub-ocr",
                    "text/html": "parse/stub-ocr",
                    "bad key!": "parse/markdown",
                }
            }
        )
    )
    assert ("stages.parse.route", "default_route_missing") in found
    assert ("stages.parse.route.text/html.component", "parser_does_not_accept") in found
    assert ("stages.parse.route.*", "invalid_route_key") in found
    assert not [path for path, _ in found if "application/pdf" in path or "scanned" in path]


def test_the_scan_stage_keeps_its_security_floor() -> None:
    found = issues(
        pipeline(scan={"checks": ["scan/dedupe"], "on_outcome": {"infected": "continue"}})
    )
    assert ("stages.scan.checks", "required_check_missing") in found
    assert ("stages.scan.on_outcome.infected", "must_exclude") in found


def test_shape_errors_name_their_path() -> None:
    found = issues(
        {
            "schema": "stratafi.pipeline.v2",
            "name": "Has Spaces",
            "stages": {"index": {"dense": None, "keyword": None}, "chunk": {"index_txt": "x"}},
        }
    )
    assert ("schema", "literal_error") in found
    assert ("name", "string_pattern_mismatch") in found
    assert ("stages.index", "value_error") in found
    assert ("stages.chunk.index_txt", "extra_forbidden") in found


def test_index_text_fields_are_checked() -> None:
    assert issues(pipeline(chunk={"index_text": "{title}\n{body}"})) == [
        ("stages.chunk.index_text", "unknown_field")
    ]


def test_a_document_that_isnt_an_object_is_refused() -> None:
    with pytest.raises(SpecInvalid) as raised:
        resolve_pipeline(["not", "an", "object"], REGISTRY)  # type: ignore[arg-type]
    assert raised.value.paths == ["(root)"]


def test_errors_never_echo_values() -> None:
    secret = "a customer's account number 12345"
    with pytest.raises(SpecInvalid) as raised:
        resolve_pipeline(
            pipeline(
                embed={"component": "embed/gateway", "params": {"alias": 5, "batch_size": secret}}
            ),
            REGISTRY,
        )
    assert secret not in repr(raised.value.issues)
    assert secret not in str(raised.value)


# --- retrieval configs ------------------------------------------------------------------------


def test_a_retrieval_config_resolves_with_defaults() -> None:
    resolved = resolve_retrieval({"schema": "stratafi.retrieval.v1", "name": "r"}, REGISTRY)
    data = dump_document(resolved)
    assert data["fusion"] == {"method": "rrf", "k": 60, "alpha": 0.5}
    assert data["rerank"] == {"component": "rerank/none", "params": {}}
    assert (data["candidates"], data["top_k"]) == (50, 8)


def test_retrieval_config_errors_name_their_path() -> None:
    with pytest.raises(SpecInvalid) as raised:
        resolve_retrieval(
            {
                "schema": "stratafi.retrieval.v1",
                "name": "r",
                "rerank": {"component": "rerank/gateway", "params": {"alias": 3}},
                "retrievers": {"dense": {"top_n": 0}},
            },
            REGISTRY,
        )
    paths = raised.value.paths
    assert "retrievers.dense.top_n" in paths
    with pytest.raises(SpecInvalid) as raised:
        resolve_retrieval(
            {"schema": "stratafi.retrieval.v1", "name": "r", "rerank": "parse/stub-ocr"},
            REGISTRY,
        )
    assert raised.value.paths == ["rerank.component"]


def test_top_k_must_fit_in_the_candidates() -> None:
    with pytest.raises(SpecInvalid) as raised:
        resolve_retrieval(
            {"schema": "stratafi.retrieval.v1", "name": "r", "candidates": 5, "top_k": 8},
            REGISTRY,
        )
    assert raised.value.issues[0].code == "value_error"


# --- routes, references and diffs -------------------------------------------------------------


def test_routes_pick_the_most_specific_match() -> None:
    resolved = resolve_pipeline(
        pipeline(
            parse={
                "route": {
                    "default": "parse/markdown",
                    "pdf[scanned]": "parse/stub-ocr",
                }
            }
        ),
        REGISTRY,
    )
    route = resolved.stages.parse.route
    assert route_for(route, "application/pdf", frozenset({"scanned"}))[0] == "pdf[scanned]"
    assert route_for(route, "application/pdf", frozenset())[0] == "default"
    assert route_for(route, "text/plain", frozenset({"scanned"}))[0] == "default"


@pytest.mark.parametrize(
    ("patterns", "media_type", "traits", "expected"),
    [
        (("*",), "anything/at-all", frozenset(), True),
        (("application/pdf",), "application/pdf", frozenset(), True),
        (("pdf",), "application/pdf", frozenset(), True),
        (("text/*",), "text/markdown", frozenset(), True),
        (("pdf[scanned]",), "application/pdf", frozenset(), False),
        (("pdf[scanned]",), "application/pdf", frozenset({"scanned"}), True),
        (("text/plain",), "text/markdown", frozenset(), False),
    ],
)
def test_accept_patterns(patterns, media_type, traits, expected) -> None:
    assert accepts(patterns, media_type, traits) is expected


def test_component_refs_list_every_component_with_its_path() -> None:
    resolved = resolve_pipeline(pipeline(), REGISTRY)
    paths = [path for path, _ in component_refs(resolved)]
    assert paths[:2] == ["stages.extract.component", "stages.extract.unpack.0"]
    assert "stages.parse.route.default" in paths
    assert "stages.index.keyword" in paths


def test_diffs_name_what_changed() -> None:
    before = dump_document(resolve_pipeline(pipeline(), REGISTRY))
    after = dump_document(
        resolve_pipeline(
            pipeline(chunk={"params": {"max_tokens": 600}}, clean=[], embed="embed/gateway"),
            REGISTRY,
        )
    )
    changes = {(change.path, change.op) for change in diff_documents(before, after)}
    assert changes == {("stages.chunk.params.max_tokens", "changed"), ("stages.clean.0", "removed")}
    assert diff_documents(before, before) == []
