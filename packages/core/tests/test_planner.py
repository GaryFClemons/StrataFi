"""The planner: stage tasks with cache keys, and where an entry stops.
Excluded documents get no downstream tasks."""

import pytest

from stratafi_core.hashing import content_hash
from stratafi_core.planner import (
    CLEAN_COMPONENT,
    SCAN_COMPONENT,
    EntryInput,
    StageOutput,
    Stop,
    TaskSpec,
    Wait,
    index_tasks,
    next_task,
    stages_until,
)
from stratafi_core.specs import resolve_pipeline
from stratafi_testing.components import stub_registry

REGISTRY = stub_registry()
ENTRY = EntryInput("policies/a.pdf", "d-1", content_hash("a"), ".pdf")
RAW = {"media_type": "application/pdf", "traits": {"pages": 3, "scanned": False}}


def spec(**stages: object):
    return resolve_pipeline(
        {"schema": "stratafi.pipeline.v1", "name": "p", "stages": stages}, REGISTRY
    )


def walk(pipeline, entry=ENTRY, *, until=None, scan_decision="continue", raw=RAW, model_env=None):
    """Plan an entry through every stage, as if each task succeeded."""
    outputs: dict[str, StageOutput] = {}
    tasks: list[TaskSpec] = []
    while True:
        step = next_task(pipeline, REGISTRY, entry, outputs, until=until, model_env=model_env)
        if isinstance(step, Stop):
            return tasks, step
        tasks.append(step)
        summary = {"extract": raw, "scan": {"decision": scan_decision}}.get(step.stage, {})
        outputs[step.stage] = StageOutput(step.cache_key, summary)


def test_an_entry_runs_every_document_stage_in_order() -> None:
    tasks, stop = walk(spec())
    assert [task.stage for task in tasks] == [
        "extract",
        "scan",
        "parse",
        "clean",
        "chunk",
        "enrich",
        "embed",
    ]
    assert stop == Stop("done")
    assert tasks[0].inputs == (ENTRY.sha256,)
    assert tasks[0].env == {"extension": ".pdf"}
    keys = {task.stage: task.cache_key for task in tasks}
    # Scan gates parse but isn't its input: parse reads the raw document.
    reads = {"scan": "extract", "parse": "extract", "clean": "parse", "chunk": "clean"}
    reads |= {"enrich": "chunk", "embed": "enrich"}
    for task in tasks[1:]:
        assert task.inputs == (keys[reads[task.stage]],), task.stage
    assert tasks[1].component_id == SCAN_COMPONENT
    assert tasks[3].component_id == CLEAN_COMPONENT
    assert tasks[4].params["index_text"] == "{title}\n{heading_path}\n{text}"


def test_excluded_documents_get_no_downstream_tasks() -> None:
    tasks, stop = walk(spec(), scan_decision="exclude")
    assert [task.stage for task in tasks] == ["extract", "scan"]
    assert stop == Stop("excluded")


def test_flagged_documents_continue() -> None:
    tasks, stop = walk(spec(), scan_decision="flag")
    assert tasks[-1].stage == "embed"
    assert stop == Stop("done")


def test_a_run_stops_at_its_until_stage() -> None:
    tasks, stop = walk(spec(), until="parse")
    assert [task.stage for task in tasks] == ["extract", "scan", "parse"]
    assert stop == Stop("done")
    assert stages_until("chunk")[-1] == "chunk"
    with pytest.raises(ValueError, match="stage"):
        stages_until("nowhere")


def test_keys_are_the_same_for_the_same_work() -> None:
    first, _ = walk(spec())
    second, _ = walk(spec(chunk={"params": {"max_tokens": 400}}))
    assert [t.cache_key for t in first] == [t.cache_key for t in second]


def test_changing_the_chunker_changes_only_chunk_and_later_keys() -> None:
    before, _ = walk(spec())
    after, _ = walk(spec(chunk={"component": "chunk/fixed-tokens"}))
    same = [a.cache_key == b.cache_key for a, b in zip(before, after, strict=True)]
    assert same == [True, True, True, True, False, False, False]


def test_bumping_a_scan_checks_version_changes_the_scan_key() -> None:
    from stratafi_core.components import ComponentRegistry
    from stratafi_testing.components import stub

    bumped = ComponentRegistry(
        [c for c in stub_registry() if c.info.id != "scan/dedupe"]
        + [stub("scan/dedupe", "scan", version="1.1.0")]
    )
    pipeline = spec()
    before = next_task(pipeline, REGISTRY, ENTRY, {"extract": StageOutput("e" * 64, RAW)})
    after = next_task(pipeline, bumped, ENTRY, {"extract": StageOutput("e" * 64, RAW)})
    assert isinstance(before, TaskSpec)
    assert isinstance(after, TaskSpec)
    assert before.cache_key != after.cache_key


def test_duplicates_and_extensions_change_their_keys() -> None:
    duplicate = EntryInput("copy.pdf", "d-1", ENTRY.sha256, ".pdf", duplicate=True)
    renamed = EntryInput("a.txt", "d-1", ENTRY.sha256, ".txt")
    original, _ = walk(spec())
    copied, _ = walk(spec(), duplicate)
    other, _ = walk(spec(), renamed)
    assert copied[0].cache_key == original[0].cache_key  # extract: same bytes, same extension
    assert copied[1].cache_key != original[1].cache_key  # scan: one is a duplicate
    assert other[0].cache_key != original[0].cache_key  # extract: the extension changed


def test_parse_follows_the_route_and_refuses_types_no_parser_takes() -> None:
    pipeline = spec(parse={"route": {"default": "parse/pypdf", "pdf[scanned]": "parse/stub-ocr"}})
    scanned = {"media_type": "application/pdf", "traits": {"scanned": True}}
    tasks, _ = walk(pipeline, raw=scanned, until="parse")
    assert (tasks[-1].component_id, tasks[-1].route) == ("parse/stub-ocr", "pdf[scanned]")
    tasks, _ = walk(pipeline, until="parse")
    assert (tasks[-1].component_id, tasks[-1].route) == ("parse/pypdf", "default")
    tasks, stop = walk(pipeline, raw={"media_type": "image/png", "traits": {}})
    assert [task.stage for task in tasks] == ["extract", "scan"]
    assert stop == Stop("unsupported_type")


def test_the_model_revision_is_part_of_the_embed_key() -> None:
    one, _ = walk(spec(), model_env={"embed": {"revision": "r1"}})
    two, _ = walk(spec(), model_env={"embed": {"revision": "r2"}})
    assert one[-1].env == {"revision": "r1"}
    assert one[-1].cache_key != two[-1].cache_key
    assert one[-2].cache_key == two[-2].cache_key


def test_index_tasks_read_every_embedded_entry() -> None:
    pairs = [(content_hash("c1"), content_hash("v1")), (content_hash("c2"), content_hash("v2"))]
    dense, keyword = index_tasks(spec(), REGISTRY, pairs)
    assert dense.component_id == "index/exact"
    assert dense.inputs == (pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1])
    assert keyword.component_id == "index/bm25"
    assert keyword.inputs == (pairs[0][0], pairs[1][0])
    only_keyword = index_tasks(spec(index={"dense": None}), REGISTRY, pairs)
    assert [task.component_id for task in only_keyword] == ["index/bm25"]


def test_a_corpus_wide_clean_rule_waits_for_the_corpus() -> None:
    pipeline = spec(clean=["clean/whitespace", "clean/near-duplicates"])
    outputs: dict[str, StageOutput] = {}
    for _ in range(3):
        step = next_task(pipeline, REGISTRY, ENTRY, outputs)
        assert isinstance(step, TaskSpec)
        summary = {"extract": RAW, "scan": {"decision": "continue"}}.get(step.stage, {})
        outputs[step.stage] = StageOutput(step.cache_key, summary)
    assert next_task(pipeline, REGISTRY, ENTRY, outputs) == Wait("corpus")
    own = outputs["parse"].cache_key
    corpus = [("b" * 64, ["2026-01-01", "", "b.pdf"]), (own, ["", "", "policies/a.pdf"])]
    clean = next_task(pipeline, REGISTRY, ENTRY, outputs, corpus=corpus)
    assert isinstance(clean, TaskSpec)
    assert clean.stage == "clean"
    assert clean.inputs == (own, "b" * 64)
    assert [item["key"] for item in clean.params["corpus"]] == sorted([own, "b" * 64])
    # Another document changing changes this one's key.
    changed = next_task(
        pipeline, REGISTRY, ENTRY, outputs, corpus=[("c" * 64, ["", "", "b.pdf"]), corpus[1]]
    )
    assert isinstance(changed, TaskSpec)
    assert changed.cache_key != clean.cache_key


def test_a_near_duplicate_stops_after_clean() -> None:
    outputs = {
        "extract": StageOutput("1" * 64, RAW),
        "scan": StageOutput("2" * 64, {"decision": "continue"}),
        "parse": StageOutput("3" * 64, {}),
        "clean": StageOutput("4" * 64, {"excluded": "near_duplicate"}),
    }
    assert next_task(spec(), REGISTRY, ENTRY, outputs) == Stop("excluded")
