"""Knowledge profiles: export a confirmed trial, import, review and download.

``POST /profiles`` exports a trial that has a dev result and a succeeded read of the test split
as a ``stratafi.knowledge-profile.v1`` version: the winning pipeline spec and retrieval config,
resolved, with the models and the evidence. ``POST /profiles/import`` takes one in, as YAML or
JSON. An owner approves or rejects a version, its author too, which the record marks as a
self-review; the export carries that review, which doesn't change the profile's hash.
Audit events carry IDs, versions and decisions.
"""

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from sqlalchemy.engine import Connection

from stratafi_api.automation import fire
from stratafi_api.deps import engine_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.pagination import Page, PageParams, decode_cursor, encode_cursor, page_params
from stratafi_api.problems import Problem, ProblemIssue, problem_responses
from stratafi_api.routes.evalruns import ChangeOut
from stratafi_api.routes.experiments import trial_documents, trial_results
from stratafi_core.analysis import guardrails
from stratafi_core.experiments import metric_estimate
from stratafi_core.logs import current_log_context
from stratafi_core.profiles import Evidence, KnowledgeProfile, dump_profile, load_profile
from stratafi_core.specs import (
    NAME_PATTERN,
    SpecInvalid,
    diff_documents,
    dump_document,
    resolve_pipeline,
    resolve_retrieval,
)
from stratafi_store import evalruns, runs
from stratafi_store import experiments as experiment_store
from stratafi_store import profiles as store
from stratafi_store.audit import record_audit_event
from stratafi_store.tracking import aliases_of

router = APIRouter(tags=["profiles"])

Name = Annotated[str, StringConstraints(pattern=NAME_PATTERN)]
MAX_CONTENT_CHARS = 500_000


class ProfileFromTrialIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: uuid.UUID
    trial_id: uuid.UUID = Field(description="A trial with a dev result and a test confirmation")
    name: Name
    description: str = Field(default="", max_length=2000)


class ProfileImportIn(BaseModel):
    """A stratafi.knowledge-profile.v1, as a document or as YAML or JSON text: one of the two."""

    model_config = ConfigDict(extra="forbid")

    document: dict[str, Any] | None = None
    content: str | None = Field(default=None, min_length=1, max_length=MAX_CONTENT_CHARS)

    @model_validator(mode="after")
    def _one(self) -> "ProfileImportIn":
        if (self.document is None) == (self.content is None):
            raise ValueError("give either document or content")
        return self


class DecisionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approved", "rejected"]


class ApprovalOut(BaseModel):
    decision: Literal["approved", "rejected"]
    decided_by: str
    decided_at: datetime
    self_review: bool = Field(default=False, description="Its author decided it")


class ProfileOut(BaseModel):
    id: uuid.UUID
    name: str
    version: int
    status: Literal["draft", "approved", "rejected"]
    source: Literal["trial", "import"]
    document: dict[str, Any] = Field(description="The stratafi.knowledge-profile.v1")
    document_hash: str = Field(description="The profile's hash; its review doesn't change it")
    experiment_id: uuid.UUID | None
    trial_id: uuid.UUID | None
    approvals: list[ApprovalOut]
    created: bool | None = Field(default=None, description="False: the latest version holds it")
    created_by: str
    created_at: datetime
    aliases: list[str] = Field(
        default_factory=list, description="The aliases that name this version"
    )

    @classmethod
    def of(
        cls,
        profile: store.StoredProfile,
        approvals: list[store.StoredApproval],
        created: bool | None = None,
        aliases: list[str] | None = None,
    ) -> "ProfileOut":
        return cls(
            id=profile.id,
            name=profile.name,
            version=profile.version,
            status=profile.status,  # type: ignore[arg-type]
            source=profile.source,  # type: ignore[arg-type]
            document=profile.document,
            document_hash=profile.document_hash,
            experiment_id=profile.experiment_id,
            trial_id=profile.trial_id,
            approvals=[
                ApprovalOut(
                    decision=item.decision,  # type: ignore[arg-type]
                    decided_by=item.decided_by,
                    decided_at=item.decided_at,
                    self_review=item.self_review,
                )
                for item in approvals
            ],
            created=created,
            created_by=profile.created_by,
            created_at=profile.created_at,
            aliases=aliases or [],
        )


class ProfileDiffOut(BaseModel):
    against: dict[str, Any] | None = Field(description="The version compared with: ID, version")
    changes: list[ChangeOut] = Field(description="Pipeline and retrieval, by path")


def _audit(
    conn: Connection, principal: Principal, action: str, profile_id: uuid.UUID, **details: Any
) -> None:
    record_audit_event(
        conn,
        workspace_id=principal.workspace.id,
        actor_kind=principal.actor_kind,
        actor_subject=principal.subject,
        action=action,
        resource_type="knowledge_profile",
        resource_id=str(profile_id),
        request_id=current_log_context().get("request_id"),
        details=details,
    )


def _profile_or_404(
    conn: Connection, principal: Principal, profile_id: uuid.UUID
) -> store.StoredProfile:
    profile = store.get_profile(conn, principal.workspace.id, profile_id)
    if profile is None:
        raise Problem("not_found", code="profile_not_found")
    return profile


def _result_ref(estimate: dict[str, Any] | None, eval_run_id: uuid.UUID) -> dict[str, Any] | None:
    if not estimate:
        return None
    return {**estimate, "eval_run_id": str(eval_run_id)}


def _models(pipeline_run: runs.Run | None, evaluation: evalruns.EvalRun) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model in (pipeline_run.manifest or {}).get("models", []) if pipeline_run else []:
        models[model["role"]] = {
            key: model.get(key) for key in ("component", "alias", "revision", "dimensions")
        }
    rerank = ((evaluation.context or {}).get("models") or {}).get("rerank")
    if rerank:
        models["rerank"] = {
            "component": rerank.get("component"),
            "alias": (rerank.get("params") or {}).get("alias"),
            "revision": rerank.get("revision"),
        }
    return models


@router.post(
    "/profiles",
    response_model=ProfileOut,
    status_code=201,
    summary="Export a confirmed trial as a knowledge profile version",
    description="The trial needs a dev result and a succeeded read of the test split. "
    "The profile holds its pipeline spec and retrieval config, the models and the evidence. 200 "
    "when the latest version is the same profile.",
    responses=problem_responses(400, 401, 403, 404, 409, 503),
)
def export_profile(
    request: Request,
    body: ProfileFromTrialIn,
    principal: Principal = Depends(require_role("editor")),
) -> JSONResponse:
    registry = request.app.state.registry
    workspace_id = principal.workspace.id
    with engine_of(request).begin() as conn:
        experiment = experiment_store.get_experiment(conn, workspace_id, body.experiment_id)
        if experiment is None:
            raise Problem("not_found", code="experiment_not_found")
        trial = experiment_store.get_trial(conn, workspace_id, experiment.id, body.trial_id)
        if trial is None:
            raise Problem("not_found", code="trial_not_found")
        if trial.status != "succeeded" or trial.eval_run_id is None:
            raise Problem("conflict", "The trial has no dev result.", code="trial_not_finished")
        reads = [
            item
            for item in experiment_store.confirmations_of(conn, experiment.id)
            if item.trial_id == trial.id
        ]
        evaluations = evalruns.eval_runs_by_id(conn, [item.eval_run_id for item in reads])
        confirmed = [
            item
            for item in reads
            if evaluations.get(item.eval_run_id) is not None
            and evaluations[item.eval_run_id].status == "succeeded"
        ]
        if not confirmed:
            raise Problem(
                "conflict",
                "Confirm the trial on the test split first.",
                code="not_confirmed",
            )
        confirmation = confirmed[-1]
        test_run = evaluations[confirmation.eval_run_id]
        documents = trial_documents(conn, workspace_id, trial, registry)
        if documents is None:
            raise Problem("conflict", code="trial_documents_missing")
        pipeline, retrieval = documents
        pipeline_run = (
            runs.get_run(conn, workspace_id, trial.pipeline_run_id)
            if trial.pipeline_run_id
            else None
        )
        spec = experiment.spec()
        objective = spec.objective
        shown = {
            "metric": objective.metric,
            "rule": objective.rule,
            "checkpoint": objective.checkpoint,
            "goal": objective.goal,
        }
        found = trial_results(conn, experiment, shown)
        guard = guardrails(found.results, goal=objective.goal, resamples=1000)
        row = next((item for item in guard["trials"] if item["ordinal"] == trial.ordinal), None)
        plan = experiment.plan
        profile = KnowledgeProfile(
            name=body.name,
            description=body.description,
            pipeline=pipeline,
            retrieval=retrieval,
            models=_models(pipeline_run, test_run),
            evidence=Evidence(
                experiment={
                    "id": str(experiment.id),
                    "name": experiment.name,
                    "document_hash": experiment.document_hash,
                },
                trial={
                    "id": str(trial.id),
                    "ordinal": trial.ordinal,
                    "values": trial.factor_values,
                },
                corpus=plan.get("corpus"),
                eval_set=plan.get("eval_set"),
                objective=shown,
                dev=_result_ref(trial.objective, trial.eval_run_id),
                test=_result_ref(
                    metric_estimate(
                        test_run.metrics, objective.rule, objective.checkpoint, objective.metric
                    ),
                    test_run.id,
                ),
                test_read=confirmation.ordinal,
                guardrails={
                    "group": row["group"],
                    "equivalent": row["equivalent"],
                    "recommended": guard["recommended"] == trial.ordinal,
                }
                if row
                else None,
            ),
            created_by=principal.subject,
            created_at=datetime.now(UTC),
        )
        stored, created = store.save_profile(
            conn,
            workspace_id=workspace_id,
            profile=profile,
            source="trial",
            created_by=principal.subject,
            experiment_id=experiment.id,
            trial_id=trial.id,
            confirmation_id=confirmation.id,
        )
        approvals = store.approvals_of(conn, stored.id)
        if created:
            _audit(
                conn,
                principal,
                "profile.export",
                stored.id,
                version=stored.version,
                trial=trial.ordinal,
            )
            fire(request, conn, principal, "target_changed", profile=stored.name)
    return JSONResponse(
        ProfileOut.of(stored, approvals, created).model_dump(mode="json"),
        status_code=201 if created else 200,
    )


@router.post(
    "/profiles/import",
    response_model=ProfileOut,
    status_code=201,
    summary="Import a knowledge profile (YAML or JSON) as a version of its name",
    description="Its documents must resolve with this installation's components. Its review "
    "doesn't come with it: a version starts as a draft here.",
    responses=problem_responses(400, 401, 403, 503),
)
def import_profile(
    request: Request,
    body: ProfileImportIn,
    principal: Principal = Depends(require_role("editor")),
) -> JSONResponse:
    registry = request.app.state.registry
    try:
        profile = load_profile(body.content if body.content is not None else body.document or {})
        resolve_pipeline(profile.pipeline, registry)
        resolve_retrieval(profile.retrieval, registry)
    except SpecInvalid as error:
        raise Problem(
            "invalid_request",
            "The profile is invalid.",
            code="invalid_profile",
            errors=error.paths,
            issues=[ProblemIssue(path=issue.path, code=issue.code) for issue in error.issues],
        ) from None
    with engine_of(request).begin() as conn:
        stored, created = store.save_profile(
            conn,
            workspace_id=principal.workspace.id,
            profile=profile.model_copy(update={"review": None}),
            source="import",
            created_by=principal.subject,
        )
        approvals = store.approvals_of(conn, stored.id)
        if created:
            _audit(
                conn,
                principal,
                "profile.import",
                stored.id,
                version=stored.version,
            )
            fire(request, conn, principal, "target_changed", profile=stored.name)
    return JSONResponse(
        ProfileOut.of(stored, approvals, created).model_dump(mode="json"),
        status_code=201 if created else 200,
    )


@router.get(
    "/profiles",
    response_model=Page[ProfileOut],
    summary="The workspace's knowledge profile versions, newest first",
    responses=problem_responses(400, 401, 403, 503),
)
def list_profiles(
    request: Request,
    page: PageParams = Depends(page_params),
    name: Annotated[str | None, Query(pattern=NAME_PATTERN)] = None,
    principal: Principal = Depends(require_role("viewer")),
) -> Page[ProfileOut]:
    after = None
    if page.cursor is not None:
        created_at, profile_ref = decode_cursor(page.cursor, size=2)
        try:
            after = (datetime.fromisoformat(created_at), uuid.UUID(profile_ref))
        except (TypeError, ValueError, AttributeError):
            raise Problem(
                "invalid_request", "The cursor isn't valid.", code="invalid_cursor"
            ) from None
    with engine_of(request).connect() as conn:
        found = store.list_profiles(
            conn, principal.workspace.id, limit=page.limit + 1, after=after, name=name
        )
        named = aliases_of(
            conn, principal.workspace.id, "knowledge_profile", [item.id for item in found]
        )
        items = [
            ProfileOut.of(item, store.approvals_of(conn, item.id), aliases=named.get(item.id))
            for item in found
        ]
    cursor = None
    if len(found) > page.limit:
        last = found[page.limit - 1]
        cursor = encode_cursor([last.created_at.isoformat(), str(last.id)])
    return Page[ProfileOut](items=items[: page.limit], next_cursor=cursor)


@router.get(
    "/profiles/{profile_id}",
    response_model=ProfileOut,
    summary="One knowledge profile version, with its review",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def get_profile(
    request: Request, profile_id: uuid.UUID, principal: Principal = Depends(require_role("viewer"))
) -> ProfileOut:
    with engine_of(request).connect() as conn:
        profile = _profile_or_404(conn, principal, profile_id)
        named = aliases_of(conn, principal.workspace.id, "knowledge_profile", [profile.id])
        return ProfileOut.of(
            profile, store.approvals_of(conn, profile.id), aliases=named.get(profile.id)
        )


@router.get(
    "/profiles/{profile_id}/export",
    summary="Download a profile version as YAML or JSON, with its review",
    responses={
        200: {"content": {"application/yaml": {}, "application/json": {}}},
        **problem_responses(400, 401, 403, 404, 503),
    },
)
def download_profile(
    request: Request,
    profile_id: uuid.UUID,
    format: Annotated[Literal["yaml", "json"], Query()] = "yaml",
    principal: Principal = Depends(require_role("viewer")),
) -> Response:
    with engine_of(request).connect() as conn:
        stored = _profile_or_404(conn, principal, profile_id)
        review = store.review_of(stored, store.approvals_of(conn, stored.id))
    profile = stored.profile().model_copy(update={"review": review})
    filename = f"{stored.name}-v{stored.version}.{format}"
    return Response(
        dump_profile(profile, format),
        media_type="application/yaml" if format == "yaml" else "application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/profiles/{profile_id}/diff",
    response_model=ProfileDiffOut,
    summary="How a profile version differs from another, by default its previous version",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def diff_profile(
    request: Request,
    profile_id: uuid.UUID,
    against: Annotated[uuid.UUID | None, Query(description="Another version's ID")] = None,
    principal: Principal = Depends(require_role("viewer")),
) -> ProfileDiffOut:
    with engine_of(request).connect() as conn:
        profile = _profile_or_404(conn, principal, profile_id)
        other = (
            _profile_or_404(conn, principal, against)
            if against is not None
            else store.previous_version(conn, profile)
        )

    def compared(item: store.StoredProfile | None) -> dict[str, Any]:
        if item is None:
            return {}
        loaded = item.profile()
        return {
            "pipeline": {**dump_document(loaded.pipeline), "name": None},
            "retrieval": {**dump_document(loaded.retrieval), "name": None},
        }

    return ProfileDiffOut(
        against={"id": str(other.id), "name": other.name, "version": other.version}
        if other
        else None,
        changes=[
            ChangeOut(path=change.path, op=change.op, before=change.before, after=change.after)
            for change in diff_documents(compared(other), compared(profile))
        ]
        if other
        else [],
    )


@router.post(
    "/profiles/{profile_id}/approvals",
    response_model=ProfileOut,
    summary="Approve or reject a profile version: an owner, its author too (a self-review)",
    responses=problem_responses(400, 401, 403, 404, 503),
)
def decide_profile(
    request: Request,
    profile_id: uuid.UUID,
    body: DecisionIn,
    principal: Principal = Depends(require_role("owner")),
) -> ProfileOut:
    with engine_of(request).begin() as conn:
        profile = _profile_or_404(conn, principal, profile_id)
        self_review = principal.subject == profile.created_by
        profile = store.decide(conn, profile, decision=body.decision, decided_by=principal.subject)
        _audit(
            conn,
            principal,
            f"profile.{body.decision}",
            profile.id,
            version=profile.version,
            self_review=self_review,
        )
        fire(request, conn, principal, "target_changed", profile=profile.name)
        return ProfileOut.of(profile, store.approvals_of(conn, profile.id))
