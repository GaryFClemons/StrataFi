"""Pipeline specs and retrieval configs: validate, save as versions, read, and diff.

Both are named chains of immutable versions (``stratafi_store.specs``). The request body of a
save or a validation is the document itself, ``stratafi.pipeline.v1`` or
``stratafi.retrieval.v1`` (``contracts/schemas``). Every document goes through the same
validator the UI, the SDK and the planner use (``stratafi_core.specs``), and an invalid one is a
400 problem whose ``issues`` give each problem's path and code, never the value.
"""

import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Body, Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from stratafi_api.automation import fire
from stratafi_api.deps import engine_of
from stratafi_api.identity import Principal, require_role
from stratafi_api.pagination import Page, PageParams, decode_cursor, encode_cursor, page_params
from stratafi_api.problems import Problem, ProblemIssue, problem_responses
from stratafi_core.components import ComponentRegistry
from stratafi_core.logs import current_log_context
from stratafi_core.specs import (
    PipelineSpec,
    RetrievalConfig,
    SpecInvalid,
    diff_documents,
    document_hash,
    dump_document,
    resolve_pipeline,
    resolve_retrieval,
    spec_hash,
)
from stratafi_store import specs as store
from stratafi_store.audit import record_audit_event


class Issue(BaseModel):
    path: str
    code: str


class Change(BaseModel):
    path: str
    op: Literal["added", "removed", "changed"]
    before: Any = None
    after: Any = None


def invalid(error: SpecInvalid) -> Problem:
    return Problem(
        "invalid_request",
        "The document is invalid.",
        code="invalid_spec",
        errors=error.paths,
        issues=[ProblemIssue(path=issue.path, code=issue.code) for issue in error.issues],
    )


class _VersionBase(BaseModel):
    id: uuid.UUID
    name: str
    version: int
    spec_hash: str
    document_hash: str
    created_by: str
    created_at: datetime

    @classmethod
    def of(cls, stored: store.StoredDocument) -> Any:
        return cls.model_validate(
            {
                "id": stored.id,
                "name": stored.name,
                "version": stored.version,
                "spec_hash": stored.spec_hash,
                "document_hash": stored.document_hash,
                "created_by": stored.created_by,
                "created_at": stored.created_at,
                "document": stored.document,
            }
        )


class PipelineSpecVersion(_VersionBase):
    document: PipelineSpec


class RetrievalConfigVersion(_VersionBase):
    document: RetrievalConfig


class _ValidationBase(BaseModel):
    valid: bool
    issues: list[Issue]
    spec_hash: str | None = None


class PipelineSpecValidation(_ValidationBase):
    document: PipelineSpec | None = None


class RetrievalConfigValidation(_ValidationBase):
    document: RetrievalConfig | None = None


class SpecDiff(BaseModel):
    name: str
    from_version: int
    to_version: int
    changes: list[Change]


def _build(
    *,
    kind: Literal["pipeline", "retrieval"],
    prefix: str,
    version_model: type[_VersionBase],
    validation_model: type[_ValidationBase],
    resolve: Callable[[Any, ComponentRegistry], Any],
    noun: str,
    tag: str,
) -> APIRouter:
    router = APIRouter(tags=[tag])
    schema_name = "stratafi.pipeline.v1" if kind == "pipeline" else "stratafi.retrieval.v1"
    VersionOut = version_model
    ValidationOut = validation_model
    DiffOut = SpecDiff
    body_description = f"A `{schema_name}` document (contracts/schemas)."
    DocumentBody = Annotated[dict[str, Any], Body(description=body_description)]

    def registry(request: Request) -> ComponentRegistry:
        return request.app.state.registry

    def get_or_404(request: Request, principal: Principal, name: str, version: int | None):
        with engine_of(request).connect() as conn:
            found = store.get_version(conn, kind, principal.workspace.id, name, version)
        if found is None:
            raise Problem("not_found", code=f"{kind}_not_found")
        return found

    @router.post(
        f"{prefix}/validate",
        name=f"validate_{noun}",
        response_model=ValidationOut,
        summary=f"Validate a {noun.replace('_', ' ')} without saving it",
        responses=problem_responses(400, 401, 403),
    )
    def validate(
        request: Request, body: DocumentBody, _: Principal = Depends(require_role("viewer"))
    ) -> ValidationOut:
        try:
            resolved = resolve(body, registry(request))
        except SpecInvalid as exc:
            return ValidationOut(
                valid=False, issues=[Issue(path=i.path, code=i.code) for i in exc.issues]
            )
        return ValidationOut(
            valid=True, issues=[], spec_hash=spec_hash(resolved), document=resolved
        )

    @router.post(
        prefix,
        name=f"save_{noun}",
        response_model=VersionOut,
        status_code=201,
        summary=f"Save a {noun.replace('_', ' ')} as its name's next version",
        description="201 with a new version; 200 with the latest version when the document "
        "matches it exactly.",
        responses={
            200: {"model": VersionOut, "description": "Unchanged: the latest version"},
            **problem_responses(400, 401, 403, 503),
        },
    )
    def save(
        request: Request, body: DocumentBody, principal: Principal = Depends(require_role("editor"))
    ) -> JSONResponse:
        try:
            resolved = resolve(body, registry(request))
        except SpecInvalid as exc:
            raise invalid(exc) from None
        with engine_of(request).begin() as conn:
            stored, created = store.save_version(
                conn,
                kind,
                workspace_id=principal.workspace.id,
                document=dump_document(resolved),
                spec_hash=spec_hash(resolved),
                document_hash=document_hash(resolved),
                created_by=principal.subject,
            )
            if created:
                record_audit_event(
                    conn,
                    workspace_id=principal.workspace.id,
                    actor_kind=principal.actor_kind,
                    actor_subject=principal.subject,
                    action=f"{noun}.save",
                    resource_type=noun,
                    resource_id=str(stored.id),
                    request_id=current_log_context().get("request_id"),
                    details={"version": stored.version},
                )
                fire(request, conn, principal, "target_changed", **{kind: stored.name})
        body_out = VersionOut.of(stored).model_dump(mode="json", by_alias=True)
        return JSONResponse(body_out, status_code=201 if created else 200)

    @router.get(
        prefix,
        name=f"list_{noun}s",
        response_model=Page[VersionOut],
        summary=f"The latest version of every {noun.replace('_', ' ')}, by name",
        responses=problem_responses(400, 401, 403, 503),
    )
    def list_latest(
        request: Request,
        page: PageParams = Depends(page_params),
        principal: Principal = Depends(require_role("viewer")),
    ) -> Page[VersionOut]:
        after = None
        if page.cursor is not None:
            (after,) = decode_cursor(page.cursor, size=1)
            if not isinstance(after, str):
                raise Problem("invalid_request", "The cursor isn't valid.", code="invalid_cursor")
        with engine_of(request).connect() as conn:
            found = store.list_latest(
                conn, kind, principal.workspace.id, limit=page.limit + 1, after_name=after
            )
        items = [VersionOut.of(item) for item in found[: page.limit]]
        cursor = encode_cursor([found[page.limit - 1].name]) if len(found) > page.limit else None
        return Page[VersionOut](items=items, next_cursor=cursor)

    @router.get(
        f"{prefix}/{{name}}",
        name=f"get_{noun}",
        response_model=VersionOut,
        summary=f"The latest version of a {noun.replace('_', ' ')}",
        responses=problem_responses(400, 401, 403, 404, 503),
    )
    def get_latest(
        request: Request, name: str, principal: Principal = Depends(require_role("viewer"))
    ) -> VersionOut:
        return VersionOut.of(get_or_404(request, principal, name, None))

    @router.get(
        f"{prefix}/{{name}}/versions",
        name=f"list_{noun}_versions",
        response_model=Page[VersionOut],
        summary=f"A {noun.replace('_', ' ')}'s versions, newest first",
        responses=problem_responses(400, 401, 403, 503),
    )
    def list_versions(
        request: Request,
        name: str,
        page: PageParams = Depends(page_params),
        principal: Principal = Depends(require_role("viewer")),
    ) -> Page[VersionOut]:
        before = None
        if page.cursor is not None:
            (before,) = decode_cursor(page.cursor, size=1)
            if not isinstance(before, int):
                raise Problem("invalid_request", "The cursor isn't valid.", code="invalid_cursor")
        with engine_of(request).connect() as conn:
            found = store.list_versions(
                conn,
                kind,
                principal.workspace.id,
                name,
                limit=page.limit + 1,
                before_version=before,
            )
        items = [VersionOut.of(item) for item in found[: page.limit]]
        cursor = encode_cursor([found[page.limit - 1].version]) if len(found) > page.limit else None
        return Page[VersionOut](items=items, next_cursor=cursor)

    @router.get(
        f"{prefix}/{{name}}/versions/{{version}}",
        name=f"get_{noun}_version",
        response_model=VersionOut,
        summary=f"One version of a {noun.replace('_', ' ')}",
        responses=problem_responses(400, 401, 403, 404, 503),
    )
    def get_version(
        request: Request,
        name: str,
        version: int,
        principal: Principal = Depends(require_role("viewer")),
    ) -> VersionOut:
        return VersionOut.of(get_or_404(request, principal, name, version))

    @router.get(
        f"{prefix}/{{name}}/diff",
        name=f"diff_{noun}_versions",
        response_model=DiffOut,
        summary="What changed between two versions",
        responses=problem_responses(400, 401, 403, 404, 503),
    )
    def diff(
        request: Request,
        name: str,
        from_version: Annotated[int, Query(ge=1)],
        to_version: Annotated[int, Query(ge=1)],
        principal: Principal = Depends(require_role("viewer")),
    ) -> DiffOut:
        before = get_or_404(request, principal, name, from_version)
        after = get_or_404(request, principal, name, to_version)
        changes = [
            Change(path=c.path, op=c.op, before=c.before, after=c.after)
            for c in diff_documents(before.document, after.document)
        ]
        return DiffOut(name=name, from_version=from_version, to_version=to_version, changes=changes)

    return router


pipeline_router = _build(
    kind="pipeline",
    prefix="/pipeline-specs",
    version_model=PipelineSpecVersion,
    validation_model=PipelineSpecValidation,
    resolve=resolve_pipeline,
    noun="pipeline_spec",
    tag="pipeline specs",
)
retrieval_router = _build(
    kind="retrieval",
    prefix="/retrieval-configs",
    version_model=RetrievalConfigVersion,
    validation_model=RetrievalConfigValidation,
    resolve=resolve_retrieval,
    noun="retrieval_config",
    tag="retrieval configs",
)
