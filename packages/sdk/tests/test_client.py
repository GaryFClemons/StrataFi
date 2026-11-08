"""The SDK's client: references, variants, waiting over server-sent events,
retries, development sign-in, uploads, and results as dicts and DataFrames."""

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest

from stratafi import ApiError, Client, ConfigError, Pipeline, UnreachableError
from stratafi.client import sse_events
from stratafi.resources import as_id, parse_ref, upload_files, variant_name
from stratafi.results import Records

RUN_ID = "00000000-0000-0000-0000-000000000001"


def client_for(api: Any, **kwargs: Any) -> Client:
    return Client("http://stratafi.test", transport=api.transport, **kwargs)


# --- configuration and HTTP ---------------------------------------------------------------


def test_from_env_reads_the_address_and_the_token(api: Any) -> None:
    api.on("GET", "/corpora", api.page())
    environ = {"STRATAFI_URL": "http://stratafi.test:8100/", "STRATAFI_TOKEN": "t0"}
    with Client.from_env(environ, transport=api.transport) as client:
        assert client.url == "http://stratafi.test:8100/api/v1"
        assert len(client.corpora.list()) == 0
    assert api.requests[0].headers["X-Stratafi-Identity"] == "t0"
    with pytest.raises(ConfigError, match="STRATAFI_URL"):
        Client.from_env({})
    assert Client("https://lab.test/api/v1").url == "https://lab.test/api/v1"


def test_development_sign_in_asks_for_tokens_and_renews_them(api: Any) -> None:
    tokens = iter(["first", "second"])
    api.on(
        "POST",
        "/dev/token",
        lambda request: httpx2.Response(
            200, json={"token": next(tokens), "expires_in": 300, "subject": "u-tenant0-editor"}
        ),
    )
    api.on("GET", "/corpora", api.page())
    client = client_for(api, dev_sign_in="tenant0:editor")
    client.corpora.list()
    client.corpora.list()
    assert api.body("POST", "/dev/token") == {"tenant": "tenant0", "role": "editor"}
    assert len(api.sent("POST", "/dev/token")) == 1
    client._dev.renew_at = 0.0  # type: ignore[union-attr]  # the token has aged
    client.corpora.list()
    identities = [request.headers["X-Stratafi-Identity"] for request in api.sent("GET", "/corpora")]
    assert identities == ["first", "first", "second"]


def test_an_api_without_development_sign_in_says_to_use_a_token(api: Any) -> None:
    client = client_for(api, dev_sign_in="tenant0:viewer")
    with pytest.raises(ConfigError, match="STRATAFI_TOKEN"):
        client.corpora.list()


def test_gets_are_retried_and_posts_are_not(api: Any) -> None:
    api.on("GET", "/corpora", httpx2.Response(503), api.page())
    client = client_for(api, token="t")
    assert len(client.corpora.list()) == 0
    assert len(api.sent("GET", "/corpora")) == 2
    api.problem("POST", "/corpora", 503)
    with pytest.raises(ApiError) as error:
        client.corpora.create("halcyon")
    assert error.value.exit_code == 3
    assert len(api.sent("POST", "/corpora")) == 1


def test_connection_failures_become_unreachable_errors() -> None:
    attempts = []

    def refuse(request: httpx2.Request) -> httpx2.Response:
        attempts.append(request)
        raise httpx2.ConnectError("refused", request=request)

    client = Client("http://stratafi.test", transport=httpx2.MockTransport(refuse))
    with pytest.raises(UnreachableError) as error:
        client.runs.get(RUN_ID)
    assert error.value.exit_code == 3
    assert len(attempts) == 3


def test_problems_name_the_status_code_and_fields_never_values(api: Any) -> None:
    body = {
        "title": "Invalid request",
        "status": 400,
        "detail": "The request is invalid.",
        "code": "invalid_document",
        "issues": [{"path": "stages.chunk.params.max_tokens", "code": "too_small"}],
        "request_id": "r-1",
    }
    api.on("POST", "/pipeline-specs", httpx2.Response(400, json=body))
    with pytest.raises(ApiError) as error:
        client_for(api).pipelines.push({"schema": "stratafi.pipeline.v1", "name": "x"})
    assert str(error.value) == (
        "400 Invalid request: The request is invalid. [invalid_document] "
        "(issues: stages.chunk.params.max_tokens: too_small) (request r-1)"
    )
    assert (error.value.exit_code, error.value.code) == (2, "invalid_document")


# --- references ---------------------------------------------------------------------------


def test_references() -> None:
    assert parse_ref("halcyon@latest", "corpus") == ("halcyon", None)
    assert parse_ref("halcyon-retrieval@12", "eval set") == ("halcyon-retrieval", 12)
    assert parse_ref("hybrid-v1", "config") == ("hybrid-v1", None)
    for bad in ("Halcyon", "halcyon@0", "halcyon@v1", "halcyon@", "", "a b"):
        with pytest.raises(ConfigError):
            parse_ref(bad, "corpus")
    assert as_id(RUN_ID.upper()) == RUN_ID
    assert as_id("halcyon") is None


def test_named_references_resolve_to_ids(api: Any) -> None:
    api.on("GET", "/corpora", api.page(api.corpus("other"), api.corpus()))
    api.on("GET", f"/corpora/{api.corpus()['id']}/versions/3", api.corpus_version(3))
    api.on("GET", f"/corpora/{api.corpus()['id']}", api.corpus(latest=None))
    api.on("GET", "/eval-sets", api.page(api.eval_set(latest=2)))
    client = client_for(api)
    assert client.corpora.resolve("halcyon@3") == api.corpus_version(3)["id"]
    assert client.corpora.resolve(RUN_ID) == RUN_ID  # an ID needs no lookup
    with pytest.raises(ConfigError, match="no frozen version"):
        client.corpora.resolve("halcyon@latest")
    with pytest.raises(ConfigError, match="has no version 4"):
        client.corpora.resolve("halcyon@4")
    assert client.eval_sets.resolve("halcyon-retrieval") == api.eval_set_version(2)["id"]
    assert client.eval_sets.resolve("halcyon-retrieval@latest") == api.eval_set_version(2)["id"]
    with pytest.raises(ConfigError, match="no eval set is called holdout"):
        client.eval_sets.resolve("holdout@1")


# --- variants -----------------------------------------------------------------------------


def base_pipeline(api: Any, pipeline: Any) -> Pipeline:
    api.on(
        "GET",
        "/pipeline-specs/baseline-v1/versions/2",
        api.spec("baseline-v1", 2, pipeline()),
    )
    return client_for(api).pipelines.get("baseline-v1@2")


def test_with_params_makes_a_named_variant(api: Any, pipeline: Any) -> None:
    base = base_pipeline(api, pipeline)
    variant = base.with_params("chunk", max_tokens=256, min_tokens=128)
    assert not variant.saved
    assert variant.name == "baseline-v1.chunk.max_tokens-256.min_tokens-128"
    chunk = variant.document["stages"]["chunk"]
    assert chunk["params"] == {"max_tokens": 256, "min_tokens": 128, "overlap": 0.1}
    assert variant.document["description"] == (
        "A variant of baseline-v1 v2: chunk: max_tokens=256, min_tokens=128"
    )
    assert base.document["stages"]["chunk"]["params"]["max_tokens"] == 800  # unchanged
    again = variant.with_params("embed", alias="embed-small")  # too long to spell out
    assert again.name == variant_name(
        "baseline-v1",
        {"chunk": {"max_tokens": 256, "min_tokens": 128}, "embed": {"alias": "embed-small"}},
    )
    assert again.name.startswith("baseline-v1.")
    assert len(again.name) == len("baseline-v1.") + 12
    assert again.document["stages"]["chunk"]["params"]["max_tokens"] == 256


def test_with_params_targets_a_component_by_its_id(api: Any, pipeline: Any) -> None:
    base = base_pipeline(api, pipeline)
    with pytest.raises(ConfigError, match="clean holds 2 components"):
        base.with_params("clean", collapse=True)
    variant = base.with_params("clean/whitespace", collapse=True)
    assert variant.document["stages"]["clean"][0]["params"] == {"collapse": True}
    assert variant.document["stages"]["clean"][1]["params"] == {}
    with pytest.raises(ConfigError, match="no component at enrich"):
        base.with_params("enrich", model="x")
    with pytest.raises(ConfigError, match="no component at parse/pypdf"):
        base.with_params("parse/pypdf", strict=True)


def test_with_params_expands_a_bare_component_id() -> None:
    document = {
        "schema": "stratafi.pipeline.v1",
        "name": "p",
        "stages": {
            "parse": {"route": {"default": "parse/markdown"}},
            "embed": "embed/gateway",
        },
    }
    variant = (
        Pipeline(document)
        .with_params("embed", alias="e")
        .with_params("parse/markdown", strict=True)
    )
    stages = variant.document["stages"]
    assert stages["embed"] == {"component": "embed/gateway", "params": {"alias": "e"}}
    assert stages["parse"]["route"]["default"]["params"] == {"strict": True}


def test_long_or_structured_changes_get_a_digest_name() -> None:
    changes = {"chunk": {"index_text": "{title}\n{text}", "max_tokens": 900}}
    long = variant_name("a-pipeline-with-a-rather-long-name-already-v12", changes)
    assert long == variant_name("a-pipeline-with-a-rather-long-name-already-v12", changes)
    assert len(long) <= 64
    assert long.startswith("a-pipeline-with-a-rather-long-name-already-v12.")
    assert variant_name("p", {"chunk": {"stops": [1, 2]}}) != variant_name(
        "p", {"chunk": {"stops": [1, 3]}}
    )


def test_running_a_variant_saves_it_first(api: Any, pipeline: Any, sse_response: Any) -> None:
    base = base_pipeline(api, pipeline)
    variant = base.with_params("chunk", max_tokens=256, min_tokens=128)
    saved = api.spec(variant.name, 1, variant.document)
    api.on("POST", "/pipeline-specs", httpx2.Response(201, json=saved))
    api.on("POST", "/runs", httpx2.Response(202, json=api.run("queued")))
    api.on("GET", f"/runs/{RUN_ID}", api.run("running"), api.run("running"), api.run("succeeded"))
    api.on(
        "GET",
        "/events",
        sse_response(
            ("job", {"job_id": "j", "status": "running"}),
            ("run", {"run_id": RUN_ID, "status": "running"}),
            ("run", {"run_id": RUN_ID, "status": "succeeded"}),
        ),
    )
    updates: list[dict[str, Any]] = []
    run = client_for(api).runs.start(
        variant, corpus=api.corpus_version(1)["id"], wait=True, sample=5, on_update=updates.append
    )
    assert run.status == "succeeded"
    assert api.body("POST", "/pipeline-specs")["name"] == variant.name
    body = api.body("POST", "/runs")
    assert body["pipeline_spec_id"] == saved["id"]
    assert body["sample"] == {"size": 5}
    assert [update["status"] for update in updates] == ["running", "succeeded"]
    assert api.sent("GET", "/events")[0].url.params["run_id"] == RUN_ID


# --- waiting ------------------------------------------------------------------------------


def test_a_finished_run_needs_no_events(api: Any) -> None:
    api.on("GET", f"/runs/{RUN_ID}", api.run("failed", error_code="stage_failed"))
    run = client_for(api).runs.get(RUN_ID).wait()
    assert (run.status, run.succeeded, run.record.error_code) == ("failed", False, "stage_failed")
    assert api.sent("GET", "/events") == []


def test_a_run_that_finishes_while_subscribing_is_seen(api: Any, sse_response: Any) -> None:
    api.on("GET", f"/runs/{RUN_ID}", api.run("running"), api.run("running"), api.run("succeeded"))
    api.on("GET", "/events", sse_response())  # no event: it finished before the stream began
    run = client_for(api).runs.get(RUN_ID)
    assert run.status == "running"
    assert run.wait().status == "succeeded"
    assert len(api.sent("GET", "/events")) == 1


def test_a_lagged_stream_reads_the_record_again(api: Any, sse_response: Any) -> None:
    api.on(
        "GET",
        f"/runs/{RUN_ID}",
        api.run("running"),
        api.run("running"),
        api.run("running"),
        api.run("cancelled"),
    )
    api.on("GET", "/events", sse_response(("lagged", {})))
    assert client_for(api).runs.get(RUN_ID).wait().status == "cancelled"


def test_a_broken_stream_reconnects(api: Any, sse_response: Any) -> None:
    streams = iter([None, sse_response(("run", {"run_id": RUN_ID, "status": "succeeded"}))])

    def events(request: httpx2.Request) -> httpx2.Response:
        answer = next(streams)
        if answer is None:
            raise httpx2.ReadError("reset", request=request)
        return answer

    api.on("GET", "/events", events)
    api.on(
        "GET",
        f"/runs/{RUN_ID}",
        api.run("running"),
        api.run("running"),
        api.run("running"),
        api.run("running"),
        api.run("succeeded"),
    )
    assert client_for(api).runs.get(RUN_ID).wait().status == "succeeded"
    assert len(api.sent("GET", "/events")) == 2


def test_an_api_without_events_is_polled(api: Any) -> None:
    api.problem("GET", "/events", 503, code="events_not_configured")
    api.on("GET", f"/runs/{RUN_ID}", api.run("running"), api.run("running"), api.run("succeeded"))
    assert client_for(api).runs.get(RUN_ID).wait().status == "succeeded"
    assert len(api.sent("GET", "/events")) == 1


def test_an_eval_run_follows_its_own_events(api: Any, sse_response: Any) -> None:
    eval_id = api.eval_run()["id"]
    running = api.eval_run("running")
    api.on("GET", f"/eval-runs/{eval_id}", running, running, running, api.eval_run("succeeded"))
    api.on(
        "GET",
        "/events",
        sse_response(
            ("run", {"run_id": eval_id, "status": "succeeded"}),  # another kind: ignored
            ("eval_run", {"run_id": RUN_ID, "status": "succeeded"}),  # another eval run
            ("eval_run", {"run_id": eval_id, "status": "succeeded"}),
        ),
    )
    found = client_for(api).evals.get(eval_id).wait()
    assert found.metrics.select("recall@5", "mrr@10") == {"recall@5": 0.75, "mrr@10": 0.5}
    assert len(api.sent("GET", f"/eval-runs/{eval_id}")) == 4  # only its own event read again


def test_server_sent_events_parse() -> None:
    lines = [
        ": connected",
        "retry: 3000",
        "",
        "event: run",
        'data: {"a":',
        "data: 1}",
        "",
        "id: 2",
        "data: not json",
        "",
        "event: lagged",
        "data: {}",
        "",
    ]
    events = list(sse_events(lines))
    assert events[0] is None  # a comment wakes the reader
    assert [(event.kind, event.data) for event in events[1:]] == [
        ("run", {"a": 1}),
        ("message", {}),
        ("lagged", {}),
    ]


# --- eval sets, evals and uploads ---------------------------------------------------------


def test_eval_sets_import_by_extension(api: Any, tmp_path: Path) -> None:
    imported = {
        "eval_set": api.eval_set(),
        "version": api.eval_set_version(1),
        "created": True,
        "report": {"items": 90, "derived_tags": 90, "assigned_splits": 90, "splits": {}},
    }
    api.on("POST", "/eval-sets/import", imported)
    source = tmp_path / "items.jsonl"
    source.write_text('{"id": "q1"}\n', encoding="utf-8")
    result = client_for(api).eval_sets.import_file(
        source, name="halcyon-retrieval", as_of="2026-11-02", test_fraction=0.25
    )
    assert result.version.item_count == 90
    body = api.body("POST", "/eval-sets/import")
    assert (body["format"], body["as_of"], body["test_fraction"]) == ("jsonl", "2026-11-02", 0.25)
    with pytest.raises(ConfigError, match=r"\.yaml, \.jsonl or \.csv"):
        client_for(api).eval_sets.import_file(tmp_path / "items.txt", name="halcyon")


def test_evals_resolve_their_inputs(api: Any) -> None:
    api.on(
        "GET",
        "/retrieval-configs/hybrid-v1/versions/2",
        api.spec("hybrid-v1", 2, {"schema": "stratafi.retrieval.v1", "name": "k"}),
    )
    api.on("GET", "/eval-sets", api.page(api.eval_set()))
    api.on("GET", f"/eval-sets/{api.eval_set()['id']}/versions/1", api.eval_set_version(1))
    api.on("POST", "/eval-runs", httpx2.Response(202, json=api.eval_run()))
    started = client_for(api).evals.run(
        RUN_ID,
        retrieval="hybrid-v1@2",
        eval_set="halcyon-retrieval@1",
        options={"seed": 3},
    )
    assert started.status == "queued"
    assert not started.metrics
    assert api.body("POST", "/eval-runs") == {
        "run_id": RUN_ID,
        "retrieval_config_id": api.spec("k", 2)["id"],
        "eval_set_version_id": api.eval_set_version(1)["id"],
        "split": "dev",
        "confirm_test_read": False,
        "options": {"seed": 3},
    }


def test_upload_names_files_by_their_paths(tmp_path: Path) -> None:
    folder = tmp_path / "corpus"
    (folder / "policies").mkdir(parents=True)
    (folder / ".git").mkdir()
    (folder / "policies" / "fees.md").write_text("x", encoding="utf-8")
    (folder / "readme.md").write_text("x", encoding="utf-8")
    (folder / ".git" / "config").write_text("x", encoding="utf-8")
    loose = tmp_path / "loose.pdf"
    loose.write_bytes(b"%PDF")
    assert [name for name, _ in upload_files([folder, loose], None)] == [
        "policies/fees.md",
        "readme.md",
        "loose.pdf",
    ]
    manifest = folder / "manifest.yaml"
    manifest.write_text("documents:\n  - {path: policies/fees.md}\n", encoding="utf-8")
    assert [name for name, _ in upload_files([], manifest)] == ["policies/fees.md"]
    manifest.write_text("documents:\n  - {path: policies/gone.md}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="isn't beside it"):
        upload_files([], manifest)
    with pytest.raises(ConfigError, match="same name"):
        upload_files([loose, loose], None)
    with pytest.raises(ConfigError, match="no files"):
        upload_files([], None)


def test_upload_sends_the_manifest_with_its_files(api: Any, tmp_path: Path) -> None:
    (tmp_path / "policies").mkdir()
    (tmp_path / "policies" / "fees.md").write_text("# Fees\n", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("documents:\n  - {path: policies/fees.md}\n", encoding="utf-8")
    api.on("GET", "/corpora", api.page(api.corpus()))
    ingested = {"source_id": RUN_ID, "files": [], "added_to_corpus": 1, "counts": {"stored": 1}}
    api.on("POST", "/documents", httpx2.Response(201, json=ingested))
    results = client_for(api).corpora.upload("halcyon", manifest=manifest)
    assert results[0].added_to_corpus == 1
    request = api.sent("POST", "/documents")[0]
    assert request.url.params["corpus_id"] == api.corpus()["id"]
    content = request.content
    assert b'name="file"; filename="policies/fees.md"' in content
    assert b'name="manifest"; filename="manifest.yaml"' in content


# --- results ------------------------------------------------------------------------------


def test_metrics_select_and_convert(api: Any) -> None:
    api.on("GET", f"/eval-runs/{api.eval_run()['id']}", api.eval_run("succeeded"))
    metrics = client_for(api).evals.get(api.eval_run()["id"]).metrics
    assert metrics.select("recall@5", checkpoint="fused") == {"recall@5": 0.7}
    assert metrics.headline() == {"hit@5": 0.75, "recall@5": 0.75, "mrr@10": 0.5, "ndcg@10": 0.55}
    with pytest.raises(KeyError, match="recall@50"):
        metrics.select("recall@50")
    frame = metrics.to_frame()
    assert list(frame.columns) == ["rule", "checkpoint", "metric", "mean", "low", "high", "n"]
    assert len(frame) == 5
    assert frame.loc[frame["checkpoint"] == "fused", "mean"].tolist() == [0.7]


def test_records_convert_to_dicts_and_flat_frames(api: Any) -> None:
    api.on("GET", "/runs", api.page(api.run("succeeded"), api.run("failed")))
    runs = client_for(api).runs.list()
    assert isinstance(runs, Records)
    assert [row["status"] for row in runs.to_dicts()] == ["succeeded", "failed"]
    assert json.loads(json.dumps(runs.to_dicts()))[0]["id"] == RUN_ID
    frame = runs.to_frame()
    assert frame["counts.tasks.succeeded"].tolist()[0] == 4
    assert len(runs[:1]) == 1


def test_report_tables(api: Any) -> None:
    rows = [
        {"max_tokens": 256, "recall@5": 0.71234, "note": "a|b"},
        {"max_tokens": 512, "recall@5": None, "extra": True},
    ]
    assert client_for(api).reports.table(rows, digits=2) == "\n".join(
        [
            "| max_tokens | recall@5 | note | extra |",
            "| --- | --- | --- | --- |",
            "| 256 | 0.71 | a\\|b | - |",
            "| 512 | - | - | true |",
        ]
    )
