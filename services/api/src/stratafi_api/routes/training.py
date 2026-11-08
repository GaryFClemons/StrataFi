"""Post-training: training sets, training runs and the model registry.

- ``POST /training-sets`` mines a training set from a succeeded eval run on the dev split
  (202): the test split is locked, so a test or ``all`` run is refused. Its records come
  back by page (``.../records``) and as exports in three formats (``.../export``).
- ``POST /training-runs`` trains a model with a ``train/`` component on a training set (202).
  Training is lab only: it's refused while ``models.in_process`` is off, and a trainer
  whose extra isn't installed can't start. Its progress is the epoch it's at, with that epoch's
  metrics; ``.../manifest`` is its record (``stratafi.training-run.v1``); ``.../cancel`` stops
  it.
- ``/trained-models`` is the registry: each model's card (``stratafi.model-card.v1``), whether
  its artifact is still stored (a purge withdraws it), withdrawal, and a zip of its files and
  card, to train or serve it elsewhere.

Audit events carry IDs, counts and formats, never text.
"""

import json
import re
import tempfile
import uuid
import zipfile
from collections.abc import Iterator
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.engine import Connection

from stratafi_api.deps import engine_of, store_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.problems import Problem, problem_responses
from stratafi_core.components import ComponentError, UnknownComponent
from stratafi_core.logs import current_log_context
from stratafi_core.training import (
    ModelManifest,
    TrainingRecord,
    TrainingRunManifest,
    TrainingSetOptions,
    export_rows,
    read_records,
)
from stratafi_store import evalruns
from stratafi_store import training as store
from stratafi_store.artifacts import ArtifactNotFound, artifact_key
from stratafi_store.audit import record_audit_event
from stratafi_store.jobs import JobQueue
from stratafi_store.tracking import aliases_of

router = APIRouter(tags=["training"])

SetStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
RunStatus = Literal["queued", "running", "succeeded", "failed", "cancelling", "cancelled"]
Kind = Literal["query-adapter", "embedder", "reranker"]
ExportFormat = Literal["sentence-transformers", "cross-encoder", "chat"]
NAME_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"
SAFE_NAME = re.compile(r"[^a-z0-9._-]+")


# --- bodies ------------------------------------------------------------------------------------


class TrainingSetIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eval_run_id: uuid.UUID = Field(description="A succeeded eval run on the dev split")
    name: str | None = Field(
        default=None, pattern=NAME_PATTERN, description="Defaults to the eval set's name"
    )
    options: TrainingSetOptions = Field(default_factory=TrainingSetOptions)


class TrainingSetOut(BaseModel):
    id: uuid.UUID
    name: str
    eval_run_id: uuid.UUID
    status: SetStatus
    options: dict[str, Any]
    cache_key: str | None = Field(description="The records' artifact")
    cached: bool = Field(description="An identical earlier set's records were reused")
    context: dict[str, Any] = Field(
        description="What it was mined from: eval set, indexes, embedder, dense index"
    )
    stats: dict[str, Any] | None = Field(
        description="Records, positives, negatives by reason, what was dropped, the leak guard"
    )
    error_code: str | None
    created_by: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @classmethod
    def of(cls, found: store.TrainingSet) -> "TrainingSetOut":
        return cls(**{key: getattr(found, key) for key in cls.model_fields})


class TrainingSetsOut(BaseModel):
    items: list[TrainingSetOut]


class TrainingRecordsOut(BaseModel):
    items: list[TrainingRecord]
    total: int
    offset: int


class TrainerIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component: str = Field(description="A train/ component, such as train/query-adapter")
    params: dict[str, Any] = Field(default_factory=dict)


class TrainingRunIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    training_set_id: uuid.UUID = Field(description="A succeeded training set")
    name: str = Field(
        pattern=NAME_PATTERN, description="The model's name; each run of it is a new version"
    )
    trainer: TrainerIn


class TrainingRunOut(BaseModel):
    id: uuid.UUID
    training_set_id: uuid.UUID
    name: str
    kind: Kind
    trainer: str
    trainer_version: str
    params: dict[str, Any] = Field(description="The trainer's parameters, resolved")
    base: dict[str, Any] = Field(description="The base model, once the run has started")
    status: RunStatus
    progress: dict[str, Any] = Field(description="The epoch it's at, with that epoch's metrics")
    metrics: dict[str, Any] | None = Field(
        description="Baseline, every epoch's loss and validation metrics, the epoch kept"
    )
    manifest_hash: str | None
    model_key: str | None
    model_cached: bool = Field(description="An identical earlier run's model was reused")
    model_id: uuid.UUID | None = Field(description="The model it registered")
    error_code: str | None
    created_by: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @classmethod
    def of(cls, found: store.TrainingRun) -> "TrainingRunOut":
        return cls(**{key: getattr(found, key) for key in cls.model_fields})


class TrainingRunsOut(BaseModel):
    items: list[TrainingRunOut]


class TrainedModelOut(BaseModel):
    id: uuid.UUID
    name: str
    version: int
    kind: Kind
    training_run_id: uuid.UUID
    model_key: str
    base: dict[str, Any]
    status: Literal["ready", "withdrawn"]
    stored: bool = Field(description="Its artifact is still stored: a purge removes it")
    card: dict[str, Any] = Field(description="Its card, stratafi.model-card.v1")
    withdrawn_by: str | None
    withdrawn_at: datetime | None
    created_by: str
    created_at: datetime
    aliases: list[str] = Field(
        default_factory=list, description="The aliases that name this version"
    )

    @classmethod
    def of(
        cls, found: store.TrainedModel, *, stored: bool, aliases: list[str] | None = None
    ) -> "TrainedModelOut":
        values = {
            key: getattr(found, key) for key in cls.model_fields if key not in ("stored", "aliases")
        }
        return cls(**values, stored=stored, aliases=aliases or [])


class TrainedModelsOut(BaseModel):
    items: list[TrainedModelOut]


# --- helpers -----------------------------------------------------------------------------------


def _queue(request: Request) -> JobQueue:
    queue = request.app.state.jobs
    if queue is None:
        raise Problem("unavailable", code="database_not_configured")
    return queue


def _audit(
    conn: Connection,
    principal: Principal,
    action: str,
    resource_type: str,
    resource_id: str,
    **details: Any,
) -> None:
    record_audit_event(
        conn,
        workspace_id=principal.workspace.id,
        actor_kind=principal.actor_kind,
        actor_subject=principal.subject,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        request_id=current_log_context().get("request_id"),
        details=details,
    )


def _set_or_404(conn: Connection, principal: Principal, set_id: uuid.UUID) -> store.TrainingSet:
    found = store.get_training_set(conn, principal.workspace.id, set_id)
    if found is None:
        raise Problem("not_found", code="training_set_not_found")
    return found


def _run_or_404(conn: Connection, principal: Principal, run_id: uuid.UUID) -> store.TrainingRun:
    found = store.get_training_run(conn, principal.workspace.id, run_id)
    if found is None:
        raise Problem("not_found", code="training_run_not_found")
    return found


def _model_or_404(
    conn: Connection, principal: Principal, model_id: uuid.UUID
) -> store.TrainedModel:
    found = store.get_model(conn, principal.workspace.id, model_id)
    if found is None:
        raise Problem("not_found", code="model_not_found")
    return found


def _records(request: Request, found: store.TrainingSet) -> list[TrainingRecord]:
    if found.status != "succeeded" or not found.cache_key:
        raise Problem(
            "conflict", "The training set hasn't been mined.", code="training_set_not_done"
        )
    try:
        data = store_of(request).get(artifact_key(found.workspace_id, found.cache_key))
    except ArtifactNotFound:
        raise Problem("conflict", code="artifact_missing") from None
    return read_records(data)


def _default_name(text: str) -> str:
    return SAFE_NAME.sub("-", text.lower()).strip("-.")[:64] or "training-set"


# --- training sets -------------------------------------------------------------------------


@router.post(
    "/training-sets",
    status_code=202,
    response_model=TrainingSetOut,
    summary="Mine a training set from a dev eval run: positives, hard negatives, traps",
    description="202 with the queued set. Positives are the chunks of the expected document "
    "versions that cover each item's evidence, from the whole index; hard negatives are what "
    "the chosen checkpoints ranked that covers no evidence, traps (avoid, must_not_return) "
    "first. Only the dev split may be mined: dev items that nearly duplicate a test query are "
    "dropped. An identical set reuses the records.",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def start_training_set(
    request: Request, body: TrainingSetIn, principal: Principal = Depends(require_role("editor"))
) -> JSONResponse:
    with engine_of(request).begin() as conn:
        evaluation = evalruns.get_eval_run(conn, principal.workspace.id, body.eval_run_id)
        if evaluation is None:
            raise Problem("not_found", code="eval_run_not_found", errors=["eval_run_id"])
        if evaluation.status != "succeeded":
            raise Problem("conflict", "The eval run hasn't succeeded.", code="eval_run_not_done")
        if evaluation.split != "dev":
            raise Problem(
                "conflict",
                "Only an eval run on the dev split can be mined: the test split stays locked.",
                code="not_dev_split",
            )
        name = body.name or _default_name(
            str(evaluation.context.get("eval_set", {}).get("name") or "training-set")
        )
        created = store.create_training_set(
            conn,
            workspace_id=principal.workspace.id,
            name=name,
            eval_run_id=evaluation.id,
            options=body.options.model_dump(mode="json"),
            created_by=principal.subject,
        )
        _queue(request).enqueue_in(
            conn,
            workspace_id=principal.workspace.id,
            kind=store.BUILD_JOB,
            payload={"training_set_id": str(created.id)},
            resource_class="cpu",
            priority=5,
            max_attempts=request.app.state.settings.worker.max_attempts,
        )
        _audit(
            conn,
            principal,
            "training_set.start",
            "training_set",
            str(created.id),
            eval_run=str(evaluation.id),
        )
    return JSONResponse(TrainingSetOut.of(created).model_dump(mode="json"), status_code=202)


@router.get(
    "/training-sets",
    response_model=TrainingSetsOut,
    summary="Training sets, newest first; one eval run's with eval_run_id",
    responses=problem_responses(400, 401, 403, 503),
)
def list_training_sets(
    request: Request,
    eval_run_id: Annotated[uuid.UUID | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    principal: Principal = Depends(require_role("viewer")),
) -> TrainingSetsOut:
    with engine_of(request).connect() as conn:
        found = store.list_training_sets(
            conn, principal.workspace.id, limit=limit, eval_run_id=eval_run_id
        )
    return TrainingSetsOut(items=[TrainingSetOut.of(item) for item in found])


@router.get(
    "/training-sets/{training_set_id}",
    response_model=TrainingSetOut,
    summary="One training set: its options, what it was mined from, and its counts",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_training_set(
    request: Request,
    training_set_id: uuid.UUID,
    principal: Principal = Depends(require_role("viewer")),
) -> TrainingSetOut:
    with engine_of(request).connect() as conn:
        return TrainingSetOut.of(_set_or_404(conn, principal, training_set_id))


@router.get(
    "/training-sets/{training_set_id}/records",
    response_model=TrainingRecordsOut,
    summary="A page of a training set's records: each item's query, positives and negatives",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def list_training_records(
    request: Request,
    training_set_id: uuid.UUID,
    split: Annotated[Literal["train", "validation", "all"], Query()] = "all",
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    principal: Principal = Depends(require_role("viewer")),
) -> TrainingRecordsOut:
    with engine_of(request).connect() as conn:
        found = _set_or_404(conn, principal, training_set_id)
    records = [r for r in _records(request, found) if split == "all" or r.split == split]
    return TrainingRecordsOut(
        items=records[offset : offset + limit], total=len(records), offset=offset
    )


@router.get(
    "/training-sets/{training_set_id}/export",
    summary="A training set as JSON Lines: sentence-transformers, cross-encoder or chat",
    description="sentence-transformers: anchor, positive, negatives (one row per positive). "
    "cross-encoder: query, passage, label (1 for positives, 0 for negatives). chat: the pinned "
    "grounded-qa-v2 prompt over gold and distractor passages, answered with the item's "
    "reference answer and its citations, or the refusal when the item has no evidence; items "
    "with evidence but no reference answer are left out. Every row names its item and split.",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"application/x-ndjson": {"schema": {"type": "string"}}}},
        **problem_responses(400, 401, 403, 404, 409, 503),
    },
)
def export_training_set(
    request: Request,
    training_set_id: uuid.UUID,
    format: Annotated[ExportFormat, Query()] = "sentence-transformers",
    split: Annotated[Literal["train", "validation", "all"], Query()] = "all",
    principal: Principal = Depends(require_role("editor")),
) -> StreamingResponse:
    with engine_of(request).begin() as conn:
        found = _set_or_404(conn, principal, training_set_id)
        _audit(
            conn,
            principal,
            "training_set.export",
            "training_set",
            str(found.id),
            format=format,
            split=split,
        )
    records = _records(request, found)
    day = str(found.context.get("day") or "")

    def lines() -> Iterator[bytes]:
        for row in export_rows(records, format, day=day, split=split):
            yield json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"

    filename = f"{found.name}-{format}-{split}.jsonl"
    return StreamingResponse(
        lines(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# --- training runs -------------------------------------------------------------------------


@router.post(
    "/training-runs",
    status_code=202,
    response_model=TrainingRunOut,
    summary="Train a model on a training set with a train/ component (lab only)",
    description="202 with the queued run. The trainer's resource class picks the workers: "
    "train/query-adapter runs on cpu; train/sentence-transformers and train/cross-encoder on "
    "gpu, with the sentence-transformers extra. A succeeded run registers its model under "
    "`name` at the next version; an identical earlier run's model is reused.",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def start_training_run(
    request: Request, body: TrainingRunIn, principal: Principal = Depends(require_role("editor"))
) -> JSONResponse:
    settings = request.app.state.settings
    if not settings.models.in_process:
        raise Problem(
            "conflict",
            "Training is lab only: models.in_process is off.",
            code="in_process_disabled",
        )
    try:
        component = request.app.state.registry.get(body.trainer.component)
    except UnknownComponent:
        raise Problem(
            "invalid_request", code="unknown_component", errors=["trainer.component"]
        ) from None
    if component.info.stage != "train":
        raise Problem("invalid_request", code="not_a_trainer", errors=["trainer.component"])
    try:
        params = component.resolve_params(body.trainer.params)
    except ValueError:
        raise Problem("invalid_request", code="invalid_params", errors=["trainer.params"]) from None
    if component.unavailable() is not None:
        raise Problem("conflict", component.unavailable(), code="extra_missing")
    with engine_of(request).begin() as conn:
        found = _set_or_404(conn, principal, body.training_set_id)
        if found.status != "succeeded":
            raise Problem(
                "conflict", "The training set hasn't been mined.", code="training_set_not_done"
            )
        if component.model_kind == "query-adapter" and "dense" not in found.context.get(
            "indexes", {}
        ):
            raise Problem(
                "conflict",
                "An adapter needs a dense index: the eval run searched none.",
                code="dense_index_missing",
            )
        created = store.create_training_run(
            conn,
            workspace_id=principal.workspace.id,
            training_set_id=found.id,
            name=body.name,
            kind=component.model_kind,
            trainer=component.info.id,
            trainer_version=component.info.version,
            params=component.dump_params(params),
            created_by=principal.subject,
        )
        job = _queue(request).enqueue_in(
            conn,
            workspace_id=principal.workspace.id,
            kind=store.TRAIN_JOB,
            payload={"training_run_id": str(created.id)},
            resource_class=component.info.resource_class,
            priority=3,
            max_attempts=settings.worker.max_attempts,
        )
        store.set_job(conn, created.id, job.id)
        _audit(
            conn,
            principal,
            "training_run.start",
            "training_run",
            str(created.id),
            training_set=str(found.id),
            trainer=component.info.id,
        )
    return JSONResponse(TrainingRunOut.of(created).model_dump(mode="json"), status_code=202)


@router.get(
    "/training-runs",
    response_model=TrainingRunsOut,
    summary="Training runs, newest first; one training set's with training_set_id",
    responses=problem_responses(400, 401, 403, 503),
)
def list_training_runs(
    request: Request,
    training_set_id: Annotated[uuid.UUID | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    principal: Principal = Depends(require_role("viewer")),
) -> TrainingRunsOut:
    with engine_of(request).connect() as conn:
        found = store.list_training_runs(
            conn, principal.workspace.id, limit=limit, training_set_id=training_set_id
        )
    return TrainingRunsOut(items=[TrainingRunOut.of(item) for item in found])


@router.get(
    "/training-runs/{training_run_id}",
    response_model=TrainingRunOut,
    summary="One training run: its trainer, progress, metrics and model",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_training_run(
    request: Request,
    training_run_id: uuid.UUID,
    principal: Principal = Depends(require_role("viewer")),
) -> TrainingRunOut:
    with engine_of(request).connect() as conn:
        return TrainingRunOut.of(_run_or_404(conn, principal, training_run_id))


@router.get(
    "/training-runs/{training_run_id}/manifest",
    summary="A training run's record (stratafi.training-run.v1), once it has succeeded",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def get_training_run_manifest(
    request: Request,
    training_run_id: uuid.UUID,
    principal: Principal = Depends(require_role("viewer")),
) -> JSONResponse:
    with engine_of(request).connect() as conn:
        run = _run_or_404(conn, principal, training_run_id)
    if run.manifest is None:
        raise Problem("conflict", "The training run hasn't finished.", code="training_run_not_done")
    record = TrainingRunManifest.model_validate(run.manifest)
    return JSONResponse(record.model_dump(mode="json", by_alias=True))


@router.post(
    "/training-runs/{training_run_id}/cancel",
    response_model=TrainingRunOut,
    summary="Cancel a training run: a queued one at once, a running one at its next step",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def cancel_training_run(
    request: Request,
    training_run_id: uuid.UUID,
    principal: Principal = Depends(require_role("editor")),
) -> TrainingRunOut:
    with engine_of(request).begin() as conn:
        run = _run_or_404(conn, principal, training_run_id)
        cancelled = store.request_cancel(conn, run)
        _audit(conn, principal, "training_run.cancel", "training_run", str(run.id))
    if run.job_id is not None and not run.finished:
        _queue(request).cancel(run.job_id, principal.workspace.id)
    return TrainingRunOut.of(cancelled)


# --- the model registry --------------------------------------------------------------------


@router.get(
    "/trained-models",
    response_model=TrainedModelsOut,
    summary="Trained models, newest first; by kind, name or status",
    responses=problem_responses(400, 401, 403, 503),
)
def list_trained_models(
    request: Request,
    kind: Annotated[Kind | None, Query()] = None,
    name: Annotated[str | None, Query(pattern=NAME_PATTERN)] = None,
    status: Annotated[Literal["ready", "withdrawn"] | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    principal: Principal = Depends(require_role("viewer")),
) -> TrainedModelsOut:
    with engine_of(request).connect() as conn:
        found = store.list_models(
            conn, principal.workspace.id, limit=limit, kind=kind, name=name, status=status
        )
        kept = store.stored(conn, principal.workspace.id, {model.model_key for model in found})
        named = aliases_of(
            conn, principal.workspace.id, "trained_model", [model.id for model in found]
        )
    return TrainedModelsOut(
        items=[
            TrainedModelOut.of(
                model, stored=model.model_key in kept, aliases=named.get(model.id, [])
            )
            for model in found
        ]
    )


@router.get(
    "/trained-models/{model_id}",
    response_model=TrainedModelOut,
    summary="One trained model, with its card (stratafi.model-card.v1)",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_trained_model(
    request: Request, model_id: uuid.UUID, principal: Principal = Depends(require_role("viewer"))
) -> TrainedModelOut:
    with engine_of(request).connect() as conn:
        model = _model_or_404(conn, principal, model_id)
        kept = store.stored(conn, principal.workspace.id, {model.model_key})
        named = aliases_of(conn, principal.workspace.id, "trained_model", [model.id])
    return TrainedModelOut.of(model, stored=bool(kept), aliases=named.get(model.id, []))


@router.post(
    "/trained-models/{model_id}/withdraw",
    response_model=TrainedModelOut,
    summary="Withdraw a trained model: new runs can't use it; its record and card stay",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def withdraw_trained_model(
    request: Request, model_id: uuid.UUID, principal: Principal = Depends(require_role("editor"))
) -> TrainedModelOut:
    with engine_of(request).begin() as conn:
        model = _model_or_404(conn, principal, model_id)
        withdrawn = store.withdraw_model(conn, model, by=principal.subject)
        kept = store.stored(conn, principal.workspace.id, {model.model_key})
        named = aliases_of(conn, principal.workspace.id, "trained_model", [model.id])
        _audit(conn, principal, "trained_model.withdraw", "trained_model", str(model.id))
    return TrainedModelOut.of(withdrawn, stored=bool(kept), aliases=named.get(model.id, []))


@router.get(
    "/trained-models/{model_id}/download",
    summary="A trained model's files and card as a zip, to use outside the lab",
    description="The zip holds `model-card.json`, `manifest.json` (each file with its SHA-256) "
    "and the model's files under `model/`: an adapter's `adapter.npy`, or a sentence-"
    "transformers directory. A withdrawn model still downloads; a purged one doesn't.",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"application/zip": {"schema": {"type": "string"}}}},
        **problem_responses(400, 401, 403, 404, 409, 503),
    },
)
def download_trained_model(
    request: Request, model_id: uuid.UUID, principal: Principal = Depends(require_role("editor"))
) -> StreamingResponse:
    files = request.app.state.model_files
    if files is None:
        raise Problem("unavailable", code="storage_not_configured")
    with engine_of(request).begin() as conn:
        model = _model_or_404(conn, principal, model_id)
        if not store.stored(conn, principal.workspace.id, {model.model_key}):
            raise Problem("conflict", "The model's artifact was purged.", code="artifact_missing")
        _audit(conn, principal, "trained_model.download", "trained_model", str(model.id))
    # Spooled to disk past 64 MiB: a fine-tuned embedder is hundreds of megabytes.
    spool = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024)  # noqa: SIM115
    try:
        manifest: ModelManifest = files.manifest(model)
        with zipfile.ZipFile(spool, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("model-card.json", json.dumps(model.card, indent=2) + "\n")
            archive.writestr(
                "manifest.json", manifest.model_dump_json(by_alias=True, indent=2) + "\n"
            )
            for file in manifest.files:
                archive.writestr(f"model/{file.path}", files.read(model, file))
    except ComponentError as error:
        spool.close()
        raise Problem("conflict", code=error.code) from None
    spool.seek(0)

    def body() -> Iterator[bytes]:
        with spool:
            while chunk := spool.read(1024 * 1024):
                yield chunk

    filename = f"{model.name}-v{model.version}.zip"
    return StreamingResponse(
        body(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
