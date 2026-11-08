"""Changes that run test suites: freezing a corpus version, a new version of a
pipeline spec or retrieval config, and a profile's new version or review each fire an event
(``stratafi_store.automation.fire_event``) in the transaction that made the change, so the runs
exist only if the change commits. Each triggered run is audited with its trigger."""

from typing import Any

from fastapi import Request
from sqlalchemy.engine import Connection

from stratafi_api.identity import Principal
from stratafi_core.logs import current_log_context
from stratafi_store.audit import record_audit_event
from stratafi_store.automation import fire_event


def fire(request: Request, conn: Connection, principal: Principal, event: str, **what: Any) -> int:
    """Run the suites ``event`` concerns; returns how many runs it requested."""
    queue = request.app.state.jobs
    if queue is None:
        return 0
    started = fire_event(
        conn,
        queue,
        workspace_id=principal.workspace.id,
        event=event,
        created_by=principal.subject,
        max_attempts=request.app.state.settings.worker.max_attempts,
        **what,
    )
    for run in started:
        record_audit_event(
            conn,
            workspace_id=principal.workspace.id,
            actor_kind=principal.actor_kind,
            actor_subject=principal.subject,
            action="test_run.trigger",
            resource_type="test_run",
            resource_id=str(run.id),
            request_id=current_log_context().get("request_id"),
            details={"trigger": event, "suite_version": run.suite_version},
        )
    return len(started)
