"""Training sets, training runs and the model registry.

- A **training set** is created ``queued`` for a succeeded dev eval run, with its mining options.
  The worker's ``training_set.build`` job mines it (``stratafi_core.training.mine``) and records
  its key, what it read (``context``) and its counts (``stats``). The records themselves, with
  their texts, are an artifact; nothing here holds text.
- A **training run** is created ``queued`` with its trainer and resolved parameters; the
  worker's ``training_run.start`` job sets its base model, reports progress each epoch, and ends
  it with its metrics, its record (``stratafi.training-run.v1``) and the key of the model it
  made. Cancelling a queued run ends it at once; a running one becomes ``cancelling`` and stops
  at its next checkpoint.
- A succeeded run **registers** its model: ``trained_models`` holds it under the run's name with
  the next version for that name, its kind, base, card and artifact key. Withdrawing a model
  stops new runs from using it; a purge that removes its artifact withdraws it too
  (``stratafi_store.purge``).
- A model's **files** are its artifact's parts, listed in its manifest (``put_model``).
  ``ModelFiles`` serves them to the processes that use a model: the registry is asked on every
  use, and each file's SHA-256 checked.
"""

import hashlib
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from stratafi_core.components import ComponentError
from stratafi_core.training import (
    ADAPTER_FILE,
    ADAPTER_FORMAT,
    BaseModelRef,
    ModelFile,
    ModelManifest,
    load_adapter,
    part_name,
)
from stratafi_store.artifacts import (
    ArtifactNotFound,
    ArtifactStore,
    ObjectInfo,
    artifact_key,
    artifact_part_key,
)
from stratafi_store.tables import (
    ACTIVE_TRAINING_RUN_STATUSES,
    artifacts,
    trained_models,
    training_runs,
    training_sets,
)

BUILD_JOB = "training_set.build"
TRAIN_JOB = "training_run.start"
FINISHED = ("succeeded", "failed", "cancelled")
PURGED_BY = "system:purge"


@dataclass(frozen=True)
class TrainingSet:
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    eval_run_id: uuid.UUID
    status: str
    options: dict[str, Any]
    cache_key: str | None
    cached: bool
    context: dict[str, Any]
    stats: dict[str, Any] | None
    error_code: str | None
    created_by: str
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @property
    def finished(self) -> bool:
        return self.status in FINISHED


@dataclass(frozen=True)
class TrainingRun:
    id: uuid.UUID
    workspace_id: uuid.UUID
    training_set_id: uuid.UUID
    name: str
    kind: str
    trainer: str
    trainer_version: str
    params: dict[str, Any]
    base: dict[str, Any]
    status: str
    progress: dict[str, Any]
    metrics: dict[str, Any] | None
    manifest: dict[str, Any] | None
    manifest_hash: str | None
    model_key: str | None
    model_cached: bool
    model_id: uuid.UUID | None
    job_id: uuid.UUID | None
    error_code: str | None
    created_by: str
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @property
    def finished(self) -> bool:
        return self.status in FINISHED


@dataclass(frozen=True)
class TrainedModel:
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    version: int
    kind: str
    training_run_id: uuid.UUID
    model_key: str
    base: dict[str, Any]
    card: dict[str, Any]
    status: str
    withdrawn_by: str | None
    withdrawn_at: datetime | None
    created_by: str
    created_at: datetime
    updated_at: datetime

    @property
    def ready(self) -> bool:
        return self.status == "ready"


def _set(row: Any) -> TrainingSet:
    return TrainingSet(**{c.name: getattr(row, c.name) for c in training_sets.columns})


def _run(row: Any) -> TrainingRun:
    return TrainingRun(**{c.name: getattr(row, c.name) for c in training_runs.columns})


def _model(row: Any) -> TrainedModel:
    return TrainedModel(**{c.name: getattr(row, c.name) for c in trained_models.columns})


# --- training sets -------------------------------------------------------------------------


def create_training_set(
    conn: Connection,
    *,
    workspace_id: uuid.UUID,
    name: str,
    eval_run_id: uuid.UUID,
    options: Mapping[str, Any],
    created_by: str,
) -> TrainingSet:
    row = conn.execute(
        sa.insert(training_sets)
        .values(
            id=uuid.uuid4(),
            workspace_id=workspace_id,
            name=name,
            eval_run_id=eval_run_id,
            options=dict(options),
            created_by=created_by,
        )
        .returning(*training_sets.columns)
    ).one()
    return _set(row)


def get_training_set(
    conn: Connection, workspace_id: uuid.UUID | None, training_set_id: uuid.UUID
) -> TrainingSet | None:
    query = sa.select(training_sets).where(training_sets.c.id == training_set_id)
    if workspace_id is not None:
        query = query.where(training_sets.c.workspace_id == workspace_id)
    row = conn.execute(query).first()
    return _set(row) if row is not None else None


def list_training_sets(
    conn: Connection,
    workspace_id: uuid.UUID,
    *,
    limit: int,
    eval_run_id: uuid.UUID | None = None,
) -> list[TrainingSet]:
    """Newest first; one eval run's, or every one."""
    query = sa.select(training_sets).where(training_sets.c.workspace_id == workspace_id)
    if eval_run_id is not None:
        query = query.where(training_sets.c.eval_run_id == eval_run_id)
    rows = conn.execute(
        query.order_by(training_sets.c.created_at.desc(), training_sets.c.id.desc()).limit(limit)
    )
    return [_set(row) for row in rows]


def begin_training_set(conn: Connection, training_set_id: uuid.UUID) -> TrainingSet | None:
    """Mark a queued (or a retried running) set running; None for a finished one."""
    row = conn.execute(
        sa.update(training_sets)
        .where(
            training_sets.c.id == training_set_id,
            training_sets.c.status.in_(("queued", "running")),
        )
        .values(status="running", started_at=sa.func.now(), updated_at=sa.func.now())
        .returning(*training_sets.columns)
    ).first()
    return _set(row) if row is not None else None


def finish_training_set(
    conn: Connection, training_set: TrainingSet, *, status: str, **values: Any
) -> TrainingSet:
    """End a set's mining: its status and whatever ``values`` sets (``cache_key``, ``cached``,
    ``context``, ``stats``, ``error_code``)."""
    row = conn.execute(
        sa.update(training_sets)
        .where(training_sets.c.id == training_set.id)
        .values(status=status, finished_at=sa.func.now(), updated_at=sa.func.now(), **values)
        .returning(*training_sets.columns)
    ).one()
    return _set(row)


# --- training runs -------------------------------------------------------------------------


def create_training_run(
    conn: Connection,
    *,
    workspace_id: uuid.UUID,
    training_set_id: uuid.UUID,
    name: str,
    kind: str,
    trainer: str,
    trainer_version: str,
    params: Mapping[str, Any],
    created_by: str,
) -> TrainingRun:
    row = conn.execute(
        sa.insert(training_runs)
        .values(
            id=uuid.uuid4(),
            workspace_id=workspace_id,
            training_set_id=training_set_id,
            name=name,
            kind=kind,
            trainer=trainer,
            trainer_version=trainer_version,
            params=dict(params),
            created_by=created_by,
        )
        .returning(*training_runs.columns)
    ).one()
    return _run(row)


def set_job(conn: Connection, training_run_id: uuid.UUID, job_id: uuid.UUID) -> None:
    conn.execute(
        sa.update(training_runs).where(training_runs.c.id == training_run_id).values(job_id=job_id)
    )


def get_training_run(
    conn: Connection, workspace_id: uuid.UUID | None, training_run_id: uuid.UUID
) -> TrainingRun | None:
    query = sa.select(training_runs).where(training_runs.c.id == training_run_id)
    if workspace_id is not None:
        query = query.where(training_runs.c.workspace_id == workspace_id)
    row = conn.execute(query).first()
    return _run(row) if row is not None else None


def list_training_runs(
    conn: Connection,
    workspace_id: uuid.UUID,
    *,
    limit: int,
    training_set_id: uuid.UUID | None = None,
) -> list[TrainingRun]:
    """Newest first; one training set's, or every one."""
    query = sa.select(training_runs).where(training_runs.c.workspace_id == workspace_id)
    if training_set_id is not None:
        query = query.where(training_runs.c.training_set_id == training_set_id)
    rows = conn.execute(
        query.order_by(training_runs.c.created_at.desc(), training_runs.c.id.desc()).limit(limit)
    )
    return [_run(row) for row in rows]


def begin_training_run(conn: Connection, training_run_id: uuid.UUID) -> TrainingRun | None:
    """Mark a queued (or a retried running) run running; None for a finished or cancelling
    one."""
    row = conn.execute(
        sa.update(training_runs)
        .where(
            training_runs.c.id == training_run_id,
            training_runs.c.status.in_(("queued", "running")),
        )
        .values(status="running", started_at=sa.func.now(), updated_at=sa.func.now())
        .returning(*training_runs.columns)
    ).first()
    return _run(row) if row is not None else None


def record_progress(
    conn: Connection, training_run_id: uuid.UUID, progress: Mapping[str, Any], **values: Any
) -> str | None:
    """A running run's progress (and ``base`` once it's known); its status after, so a
    cancellation can be noticed. None when the run is gone."""
    row = conn.execute(
        sa.update(training_runs)
        .where(
            training_runs.c.id == training_run_id,
            training_runs.c.status.in_(ACTIVE_TRAINING_RUN_STATUSES),
        )
        .values(progress=dict(progress), updated_at=sa.func.now(), **values)
        .returning(training_runs.c.status)
    ).first()
    return row.status if row is not None else None


def request_cancel(conn: Connection, run: TrainingRun) -> TrainingRun:
    """Cancel a queued run at once; ask a running one to stop at its next checkpoint."""
    if run.status == "queued":
        values: dict[str, Any] = {"status": "cancelled", "finished_at": sa.func.now()}
    elif run.status == "running":
        values = {"status": "cancelling"}
    else:
        return run
    row = conn.execute(
        sa.update(training_runs)
        .where(training_runs.c.id == run.id, training_runs.c.status == run.status)
        .values(updated_at=sa.func.now(), **values)
        .returning(*training_runs.columns)
    ).first()
    if row is None:  # it moved on meanwhile
        found = get_training_run(conn, run.workspace_id, run.id)
        return found or run
    return _run(row)


def finish_training_run(
    conn: Connection, run: TrainingRun, *, status: str, **values: Any
) -> TrainingRun:
    """End a run: its status and whatever ``values`` sets (``metrics``, ``manifest``,
    ``manifest_hash``, ``model_key``, ``model_cached``, ``model_id``, ``base``,
    ``error_code``)."""
    row = conn.execute(
        sa.update(training_runs)
        .where(training_runs.c.id == run.id)
        .values(status=status, finished_at=sa.func.now(), updated_at=sa.func.now(), **values)
        .returning(*training_runs.columns)
    ).one()
    return _run(row)


# --- the model registry --------------------------------------------------------------------


def register_model(
    conn: Connection,
    *,
    run: TrainingRun,
    model_key: str,
    base: Mapping[str, Any],
    card: Callable[[uuid.UUID, int, datetime], Mapping[str, Any]],
    created_by: str,
) -> TrainedModel:
    """The run's model under its name, at the next version for that name. ``card`` makes the
    model card from the new model's ID, version and creation time."""
    # One registration per name at a time, so two runs of one name get two versions.
    conn.execute(
        sa.text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
        {"key": f"stratafi.trained_models:{run.workspace_id}:{run.name}"},
    )
    latest = conn.execute(
        sa.select(sa.func.max(trained_models.c.version)).where(
            trained_models.c.workspace_id == run.workspace_id,
            trained_models.c.name == run.name,
        )
    ).scalar_one()
    version = int(latest or 0) + 1
    model_id = uuid.uuid4()
    now = conn.execute(sa.select(sa.func.now())).scalar_one()
    row = conn.execute(
        sa.insert(trained_models)
        .values(
            id=model_id,
            workspace_id=run.workspace_id,
            name=run.name,
            version=version,
            kind=run.kind,
            training_run_id=run.id,
            model_key=model_key,
            base=dict(base),
            card=dict(card(model_id, version, now)),
            created_by=created_by,
            created_at=now,
            updated_at=now,
        )
        .returning(*trained_models.columns)
    ).one()
    return _model(row)


def get_model(
    conn: Connection, workspace_id: uuid.UUID | None, model_id: uuid.UUID
) -> TrainedModel | None:
    query = sa.select(trained_models).where(trained_models.c.id == model_id)
    if workspace_id is not None:
        query = query.where(trained_models.c.workspace_id == workspace_id)
    row = conn.execute(query).first()
    return _model(row) if row is not None else None


def list_models(
    conn: Connection,
    workspace_id: uuid.UUID,
    *,
    limit: int,
    kind: str | None = None,
    name: str | None = None,
    status: str | None = None,
) -> list[TrainedModel]:
    """Newest first, filtered by kind, name or status."""
    query = sa.select(trained_models).where(trained_models.c.workspace_id == workspace_id)
    if kind is not None:
        query = query.where(trained_models.c.kind == kind)
    if name is not None:
        query = query.where(trained_models.c.name == name)
    if status is not None:
        query = query.where(trained_models.c.status == status)
    rows = conn.execute(
        query.order_by(trained_models.c.created_at.desc(), trained_models.c.id.desc()).limit(limit)
    )
    return [_model(row) for row in rows]


def withdraw_model(conn: Connection, model: TrainedModel, *, by: str) -> TrainedModel:
    """Stop new runs from using a model; its record and card stay."""
    if not model.ready:
        return model
    row = conn.execute(
        sa.update(trained_models)
        .where(trained_models.c.id == model.id, trained_models.c.status == "ready")
        .values(
            status="withdrawn",
            withdrawn_by=by,
            withdrawn_at=sa.func.now(),
            updated_at=sa.func.now(),
        )
        .returning(*trained_models.columns)
    ).first()
    return _model(row) if row is not None else model


def withdraw_purged(conn: Connection, workspace_id: uuid.UUID, keys: set[str]) -> int:
    """Withdraw the models whose artifacts a purge removes; how many."""
    if not keys:
        return 0
    result = conn.execute(
        sa.update(trained_models)
        .where(
            trained_models.c.workspace_id == workspace_id,
            trained_models.c.model_key.in_(sorted(keys)),
            trained_models.c.status == "ready",
        )
        .values(
            status="withdrawn",
            withdrawn_by=PURGED_BY,
            withdrawn_at=sa.func.now(),
            updated_at=sa.func.now(),
        )
    )
    return result.rowcount or 0


def stored(conn: Connection, workspace_id: uuid.UUID, keys: set[str]) -> set[str]:
    """Which of ``keys`` still have their artifacts (a purge removes them)."""
    if not keys:
        return set()
    rows = conn.execute(
        sa.select(artifacts.c.cache_key).where(
            artifacts.c.workspace_id == workspace_id, artifacts.c.cache_key.in_(sorted(keys))
        )
    )
    return {row.cache_key for row in rows}


# --- a model's files -----------------------------------------------------------------------


def put_model(
    store: ArtifactStore,
    workspace_id: uuid.UUID,
    cache_key: str,
    directory: Path,
    paths: Sequence[Path],
    *,
    kind: str,
    format: str,
    base: BaseModelRef,
) -> tuple[ModelManifest, int, ObjectInfo]:
    """Store a model's files as the parts of artifact ``cache_key``, then its manifest as the
    artifact itself; the manifest, the bytes stored and the manifest's object."""
    files, total = [], 0
    for index, path in enumerate(paths):
        part = part_name(index)
        with path.open("rb") as handle:
            info = store.put(artifact_part_key(workspace_id, cache_key, part), handle)
        files.append(
            ModelFile(
                path=path.relative_to(directory).as_posix(),
                part=part,
                sha256=info.sha256,
                size=info.size,
            )
        )
        total += info.size
    manifest = ModelManifest(kind=kind, format=format, base=base, files=files)  # type: ignore[arg-type]
    data = manifest.model_dump_json(by_alias=True).encode()
    info = store.put(artifact_key(workspace_id, cache_key), data, media_type="application/json")
    return manifest, total + info.size, info


class ModelFiles:
    """Trained models' files for the processes that use them. Every use asks the
    registry first, so a withdrawn model or a purged artifact fails even when a copy of the
    model is already loaded; every file is checked against the manifest's SHA-256."""

    def __init__(self, engine: Engine, store: ArtifactStore, *, cached: int = 8) -> None:
        self.engine = engine
        self.store = store
        self.size = cached
        self._adapters: OrderedDict[tuple[uuid.UUID, str], Any] = OrderedDict()
        self._lock = threading.Lock()

    def ready(
        self, workspace_id: uuid.UUID, model_id: str, kind: str | None = None
    ) -> TrainedModel:
        """A model that may be used now; a ``ComponentError`` with the reason otherwise."""
        try:
            parsed = uuid.UUID(str(model_id))
        except ValueError:
            raise ComponentError("model_not_found") from None
        with self.engine.connect() as conn:
            model = get_model(conn, workspace_id, parsed)
            if model is None:
                raise ComponentError("model_not_found")
            if kind is not None and model.kind != kind:
                raise ComponentError("model_kind_mismatch")
            if not model.ready:
                raise ComponentError("model_withdrawn")
            if not stored(conn, workspace_id, {model.model_key}):
                raise ComponentError("artifact_missing")
        return model

    def manifest(self, model: TrainedModel) -> ModelManifest:
        try:
            data = self.store.get(artifact_key(model.workspace_id, model.model_key))
        except ArtifactNotFound:
            raise ComponentError("artifact_missing") from None
        return ModelManifest.model_validate_json(data)

    def read(self, model: TrainedModel, file: ModelFile) -> bytes:
        try:
            data = self.store.get(artifact_part_key(model.workspace_id, model.model_key, file.part))
        except ArtifactNotFound:
            raise ComponentError("artifact_missing") from None
        if hashlib.sha256(data).hexdigest() != file.sha256:
            raise ComponentError("model_corrupt")
        return data

    def adapter(self, workspace_id: uuid.UUID, model_id: str) -> tuple[TrainedModel, Any]:
        """A query adapter's weights, ``[d, d]`` float32; loaded once per process."""
        model = self.ready(workspace_id, model_id, "query-adapter")
        key = (workspace_id, model.model_key)
        with self._lock:
            if key in self._adapters:
                self._adapters.move_to_end(key)
                return model, self._adapters[key]
        manifest = self.manifest(model)
        file = next((f for f in manifest.files if f.path == ADAPTER_FILE), None)
        if manifest.format != ADAPTER_FORMAT or file is None:
            raise ComponentError("model_corrupt")
        weights = load_adapter(self.read(model, file))
        with self._lock:
            self._adapters[key] = weights
            while len(self._adapters) > self.size:
                self._adapters.popitem(last=False)
        return model, weights

    def materialize(
        self, workspace_id: uuid.UUID, model_id: str, kind: str, directory: Path
    ) -> tuple[TrainedModel, ModelManifest]:
        """Write a model's files under ``directory``, checked, to load it from there."""
        model = self.ready(workspace_id, model_id, kind)
        manifest = self.manifest(model)
        root = directory.resolve()
        for file in manifest.files:
            target = (root / file.path).resolve()
            if not target.is_relative_to(root):
                raise ComponentError("model_corrupt")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.read(model, file))
        return model, manifest


@dataclass(frozen=True)
class WorkspaceModels:
    """One workspace's trained models: ``services["trained_models"]`` for its components."""

    files: ModelFiles
    workspace_id: uuid.UUID

    def ready(self, model_id: str, kind: str | None = None) -> TrainedModel:
        return self.files.ready(self.workspace_id, model_id, kind)

    def adapter(self, model_id: str) -> tuple[TrainedModel, Any]:
        return self.files.adapter(self.workspace_id, model_id)

    def materialize(
        self, model_id: str, kind: str, directory: Path
    ) -> tuple[TrainedModel, ModelManifest]:
        return self.files.materialize(self.workspace_id, model_id, kind, directory)
