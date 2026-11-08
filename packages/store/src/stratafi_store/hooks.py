"""What a finished run or eval run tells: the experiments and the test runs that
wait on it. The run and eval-run orchestrators take this one hook as their ``on_finish``."""

import uuid

from sqlalchemy.engine import Connection

from stratafi_store import experiments, testruns
from stratafi_store.jobs import JobQueue
from stratafi_store.runs import FinishHook


def finish_hooks(queue: JobQueue, *, max_attempts: int = 3) -> FinishHook:
    hooks = (
        experiments.finish_hook(queue, max_attempts=max_attempts),
        testruns.finish_hook(queue, max_attempts=max_attempts),
    )

    def on_finish(conn: Connection, kind: str, workspace_id: uuid.UUID, run_id: uuid.UUID) -> None:
        for hook in hooks:
            hook(conn, kind, workspace_id, run_id)

    return on_finish
