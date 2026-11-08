"""Model comparison and the registries' aliases.

- ``GET /model-comparisons``: for an eval set version and a role (embedder, reranker or
  adapter), each model's best eval run with its interval, latency and egress; the controlled
  pairs (eval runs that differ only in that model), each with its paired difference; and the
  model x model matrix of those differences.
- Aliases name one version of a trained model or a knowledge profile (``champion``), one per
  name and alias: ``PUT`` sets or moves one, ``DELETE`` removes it, ``by-alias`` resolves it.
  Editors set models' aliases; a profile's alias says what's proposed for production, so it
  needs an owner. Audit events carry IDs and the alias, which is a token, never text.
"""

import re
import uuid
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Path, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.engine import Connection

from stratafi_api.deps import engine_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.problems import Problem, problem_responses
from stratafi_api.tracked import tracked_run
from stratafi_core.logs import current_log_context
from stratafi_core.model_comparison import (
    controlled_pairs,
    leaderboard,
    matrix,
    model_of,
    pair_result,
)
from stratafi_core.stats import FEW_QUERIES
from stratafi_store import evalruns, profiles, training
from stratafi_store import tracking as store
from stratafi_store.audit import record_audit_event

router = APIRouter(tags=["models"])

Role = Literal["embed", "rerank", "adapter"]
Rule = Literal["knowledge-hit-v1", "evidence-coverage-v1", "doc-v1"]
Checkpoint = Literal["dense", "keyword", "fused", "candidates", "reranked", "final"]
METRIC_RE = re.compile(r"(hit|recall|precision|mrr|ndcg)@(\d{1,3}|all)")
ALIAS_PATTERN = r"^[a-z][a-z0-9_-]{0,31}$"
NAME_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"


class ModelRowOut(BaseModel):
    key: str = Field(description="role|component|name|revision")
    label: str
    role: str
    component: str
    name: str = Field(description="A gateway alias, a catalog model or a trained model")
    revision: str | None
    runs: int
    best_eval_run_id: uuid.UUID | None
    best: dict[str, Any] | None = Field(description="The best run's mean, low, high and n")
    latency_p95_ms: float | None = Field(description="The median of its runs' p95 latency")
    egress: str | None


class ModelPairOut(BaseModel):
    a_eval_run_id: uuid.UUID
    b_eval_run_id: uuid.UUID
    a_model: str
    b_model: str
    result: dict[str, Any] | None = Field(
        description="B against A on shared queries: means, delta with its interval, p-value, "
        "wins, losses and ties"
    )


class MatrixCellOut(BaseModel):
    delta: float = Field(description="The column's model minus the row's, averaged over pairs")
    pairs: int


class ModelComparisonOut(BaseModel):
    eval_set_version_id: uuid.UUID
    role: str
    metric: str
    rule: str
    checkpoint: str
    split: str | None
    eval_runs: int = Field(description="Succeeded eval runs considered")
    models: list[ModelRowOut] = Field(description="Best first")
    pairs: list[ModelPairOut]
    matrix: list[list[MatrixCellOut | None]] = Field(
        description="matrix[i][j]: model j against model i, in the models' order"
    )
    warnings: list[str] = Field(
        description="no_controlled_pairs: no two runs differ only in the model; few_queries: "
        "a pair shares fewer than 50 queries"
    )


def _estimate(summary: dict[str, Any] | None, rule: str, checkpoint: str, metric: str):
    estimate = (((summary or {}).get("metrics") or {}).get(rule) or {}).get(checkpoint, {})
    found = estimate.get(metric) if isinstance(estimate, dict) else None
    return found if isinstance(found, dict) and "mean" in found else None


@router.get(
    "/model-comparisons",
    response_model=ModelComparisonOut,
    summary="Compare embedders, rerankers or adapters across an eval set version's eval runs",
    description="Each model's best eval run with its interval, latency and egress; controlled "
    "pairs (runs that differ only in the model) with their paired differences; and the model "
    "x model matrix of those differences.",
    responses=problem_responses(400, 401, 403, 503),
)
def compare_models(
    request: Request,
    eval_set_version_id: Annotated[uuid.UUID, Query()],
    role: Annotated[Role, Query(description="embed, rerank or adapter")] = "embed",
    metric: Annotated[str, Query(description="Such as recall@5")] = "recall@5",
    rule: Annotated[Rule, Query()] = "knowledge-hit-v1",
    checkpoint: Annotated[Checkpoint, Query()] = "final",
    split: Annotated[Literal["dev", "test", "all"] | None, Query()] = None,
    resamples: Annotated[int, Query(ge=100, le=20_000)] = 2000,
    seed: Annotated[int, Query(ge=0, le=2**31 - 1)] = 0,
    principal: Principal = Depends(require_role("viewer")),
) -> ModelComparisonOut:
    if not METRIC_RE.fullmatch(metric):
        raise Problem("invalid_request", code="unknown_metric", errors=["metric"])
    with engine_of(request).connect() as conn:
        rows = [
            row
            for row in store.search_rows(
                conn,
                principal.workspace.id,
                eval_set_version_id=eval_set_version_id,
                limit=500,
                slices=False,
            )
            if row.status == "succeeded" and (split is None or row.split == split)
        ]
        tracked = [tracked_run(row) for row in rows]
        estimates = {str(row.id): _estimate(row.metrics, rule, checkpoint, metric) for row in rows}
        board = leaderboard(
            tracked,
            role,
            metric,
            estimates=estimates,
            egress=lambda run: run.params.get(f"models.{role}.egress"),  # type: ignore[arg-type,return-value]
        )
        pairs = controlled_pairs(tracked, role)
        wanted = {tracked[index].id for pair in pairs for index in pair}
        scores = {
            run_id: {
                result.item_id: result.score
                for result in evalruns.query_results(conn, uuid.UUID(run_id))
            }
            for run_id in wanted
        }
    measured = []
    warnings: list[str] = [] if pairs else ["no_controlled_pairs"]
    for i, j in pairs:
        a, b = tracked[i], tracked[j]
        model_a, model_b = model_of(a, role), model_of(b, role)
        if model_a is None or model_b is None:  # pragma: no cover - pairs have both
            continue
        result = pair_result(
            scores[a.id],
            scores[b.id],
            rule=rule,
            checkpoint=checkpoint,
            metric=metric,
            resamples=resamples,
            seed=seed,
        )
        if result is not None and result["n"] < FEW_QUERIES and "few_queries" not in warnings:
            warnings.append("few_queries")
        measured.append(
            {
                "a_eval_run_id": a.id,
                "b_eval_run_id": b.id,
                "a_model": model_a.key,
                "b_model": model_b.key,
                "result": result,
            }
        )
    keys = [row["key"] for row in board]
    return ModelComparisonOut(
        eval_set_version_id=eval_set_version_id,
        role=role,
        metric=metric,
        rule=rule,
        checkpoint=checkpoint,
        split=split,
        eval_runs=len(rows),
        models=[ModelRowOut(**row) for row in board],
        pairs=[ModelPairOut(**pair) for pair in measured],
        matrix=[
            [MatrixCellOut(**cell) if cell else None for cell in line]
            for line in matrix(keys, measured)
        ],
        warnings=warnings,
    )


# --- aliases ------------------------------------------------------------------------------

AliasKind = Literal["trained_model", "knowledge_profile"]


class AliasOut(BaseModel):
    kind: AliasKind
    name: str
    alias: str
    target_id: uuid.UUID = Field(description="The trained model or profile version it names")
    version: int | None = Field(description="That version's number")
    moved_from: uuid.UUID | None = Field(
        default=None, description="The version it named before, when setting it moved it"
    )


class AliasesOut(BaseModel):
    items: list[AliasOut]


AliasParam = Annotated[str, Path(pattern=ALIAS_PATTERN, description="Such as champion")]
NameParam = Annotated[str, Path(pattern=NAME_PATTERN)]


def _audit(
    conn: Connection, principal: Principal, action: str, kind: str, target: uuid.UUID, alias: str
) -> None:
    record_audit_event(
        conn,
        workspace_id=principal.workspace.id,
        actor_kind=principal.actor_kind,
        actor_subject=principal.subject,
        action=action,
        resource_type=kind,
        resource_id=str(target),
        request_id=current_log_context().get("request_id"),
        details={"alias": alias},
    )


def _model(conn: Connection, principal: Principal, model_id: uuid.UUID):
    model = training.get_model(conn, principal.workspace.id, model_id)
    if model is None:
        raise Problem("not_found", code="model_not_found")
    return model


def _profile(conn: Connection, principal: Principal, profile_id: uuid.UUID):
    profile = profiles.get_profile(conn, principal.workspace.id, profile_id)
    if profile is None:
        raise Problem("not_found", code="profile_not_found")
    return profile


def _set_and_audit(
    request: Request, principal: Principal, kind: AliasKind, target: Any, alias: str
) -> AliasOut:
    with engine_of(request).begin() as conn:
        found, moved = store.set_alias(
            conn,
            principal.workspace.id,
            kind=kind,
            name=target.name,
            alias=alias,
            target_id=target.id,
            set_by=principal.subject,
        )
        _audit(conn, principal, f"{kind}.alias", kind, target.id, alias)
    return AliasOut(
        kind=kind,
        name=found.name,
        alias=found.alias,
        target_id=found.target_id,
        version=target.version,
        moved_from=moved,
    )


def _remove_and_audit(
    request: Request, principal: Principal, kind: AliasKind, target: Any, alias: str
):
    with engine_of(request).begin() as conn:
        current = store.resolve_alias(
            conn, principal.workspace.id, kind=kind, name=target.name, alias=alias
        )
        if current != target.id:
            raise Problem("not_found", code="alias_not_found")
        store.delete_alias(conn, principal.workspace.id, kind=kind, name=target.name, alias=alias)
        _audit(conn, principal, f"{kind}.unalias", kind, target.id, alias)


@router.put(
    "/trained-models/{model_id}/aliases/{alias}",
    response_model=AliasOut,
    summary="Name this version of a trained model with an alias, moving it if it named another",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def set_model_alias(
    request: Request,
    model_id: uuid.UUID,
    alias: AliasParam,
    principal: Principal = Depends(require_role("editor")),
) -> AliasOut:
    with engine_of(request).connect() as conn:
        model = _model(conn, principal, model_id)
    return _set_and_audit(request, principal, "trained_model", model, alias)


@router.delete(
    "/trained-models/{model_id}/aliases/{alias}",
    status_code=204,
    summary="Remove an alias from this version of a trained model",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def remove_model_alias(
    request: Request,
    model_id: uuid.UUID,
    alias: AliasParam,
    principal: Principal = Depends(require_role("editor")),
) -> None:
    with engine_of(request).connect() as conn:
        model = _model(conn, principal, model_id)
    _remove_and_audit(request, principal, "trained_model", model, alias)


@router.get(
    "/trained-models/by-alias/{name}/{alias}",
    response_model=AliasOut,
    summary="The trained model version an alias names",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def resolve_model_alias(
    request: Request,
    name: NameParam,
    alias: AliasParam,
    principal: Principal = Depends(require_role("viewer")),
) -> AliasOut:
    with engine_of(request).connect() as conn:
        target = store.resolve_alias(
            conn, principal.workspace.id, kind="trained_model", name=name, alias=alias
        )
        model = training.get_model(conn, principal.workspace.id, target) if target else None
    if model is None:
        raise Problem("not_found", code="alias_not_found")
    return AliasOut(
        kind="trained_model", name=name, alias=alias, target_id=model.id, version=model.version
    )


@router.put(
    "/profiles/{profile_id}/aliases/{alias}",
    response_model=AliasOut,
    summary="Name this profile version with an alias, moving it if it named another (owners)",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def set_profile_alias(
    request: Request,
    profile_id: uuid.UUID,
    alias: AliasParam,
    principal: Principal = Depends(require_role("owner")),
) -> AliasOut:
    with engine_of(request).connect() as conn:
        profile = _profile(conn, principal, profile_id)
    return _set_and_audit(request, principal, "knowledge_profile", profile, alias)


@router.delete(
    "/profiles/{profile_id}/aliases/{alias}",
    status_code=204,
    summary="Remove an alias from this profile version (owners)",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def remove_profile_alias(
    request: Request,
    profile_id: uuid.UUID,
    alias: AliasParam,
    principal: Principal = Depends(require_role("owner")),
) -> None:
    with engine_of(request).connect() as conn:
        profile = _profile(conn, principal, profile_id)
    _remove_and_audit(request, principal, "knowledge_profile", profile, alias)


@router.get(
    "/profiles/by-alias/{name}/{alias}",
    response_model=AliasOut,
    summary="The profile version an alias names",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def resolve_profile_alias(
    request: Request,
    name: NameParam,
    alias: AliasParam,
    principal: Principal = Depends(require_role("viewer")),
) -> AliasOut:
    with engine_of(request).connect() as conn:
        target = store.resolve_alias(
            conn, principal.workspace.id, kind="knowledge_profile", name=name, alias=alias
        )
        profile = profiles.get_profile(conn, principal.workspace.id, target) if target else None
    if profile is None:
        raise Problem("not_found", code="alias_not_found")
    return AliasOut(
        kind="knowledge_profile",
        name=name,
        alias=alias,
        target_id=profile.id,
        version=profile.version,
    )


@router.get(
    "/registry-aliases",
    response_model=AliasesOut,
    summary="Every alias in the workspace's registries, by kind and name",
    responses=problem_responses(400, 401, 403, 503),
)
def list_registry_aliases(
    request: Request,
    kind: Annotated[AliasKind | None, Query()] = None,
    name: Annotated[str | None, Query(pattern=NAME_PATTERN)] = None,
    principal: Principal = Depends(require_role("viewer")),
) -> AliasesOut:
    with engine_of(request).connect() as conn:
        found = store.list_aliases(conn, principal.workspace.id, kind=kind, name=name)
        versions: dict[uuid.UUID, int] = {}
        for item in found:
            getter = training.get_model if item.kind == "trained_model" else profiles.get_profile
            target = getter(conn, principal.workspace.id, item.target_id)
            if target is not None:
                versions[item.target_id] = target.version
    return AliasesOut(
        items=[
            AliasOut(
                kind=item.kind,  # type: ignore[arg-type]
                name=item.name,
                alias=item.alias,
                target_id=item.target_id,
                version=versions.get(item.target_id),
            )
            for item in found
        ]
    )
