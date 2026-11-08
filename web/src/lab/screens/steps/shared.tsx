// What every step inspector shares: a run's tasks for a stage, an artifact's
// JSON content, and how a task's status reads. Document and chunk text is
// shown as plain text only.

import type { components } from "../../api";
import { unwrap, useLoad } from "../../api/load";
import type { RunOut } from "../../components/live";
import { Empty, ErrorNote, Loading, StatusChip } from "../../components/ui";
import { useLab } from "../../context";
import type { Stage } from "../../spec/editing";

export type TaskOut = components["schemas"]["TaskOut"];
export type Summary = Record<string, unknown>;

export function summaryOf(task: TaskOut): Summary {
  return (task.summary ?? {}) as Summary;
}

export function value(summary: Summary, key: string): string {
  const found = summary[key];
  return found === undefined || found === null ? "–" : String(found);
}

/** A run's tasks for one stage, reloaded as the stage's counts change. */
export function useStageTasks(run: RunOut, stage: Stage) {
  const { api } = useLab();
  const version = JSON.stringify(run.counts.stages?.[stage] ?? {});
  return useLoad(
    async () =>
      unwrap(
        await api.GET("/runs/{run_id}/tasks", {
          params: { path: { run_id: run.id }, query: { stage, limit: 200 } },
        }),
      ).items,
    `${run.id}:${stage}:${version}`,
  );
}

/** A run's entries: names, hashes and states, reloaded as entry states change. */
export function useRunEntries(run: RunOut) {
  const { api } = useLab();
  const version = JSON.stringify(run.counts.entries ?? {});
  return useLoad(
    async () =>
      unwrap(
        await api.GET("/runs/{run_id}/entries", {
          params: { path: { run_id: run.id }, query: { limit: 200 } },
        }),
      ).items,
    `${run.id}:entries:${version}`,
  );
}

/** An artifact's JSON content: a raw document, scan report, or parsed document. */
export function useArtifactJson<T>(cacheKey: string | null) {
  const { api } = useLab();
  return useLoad<T | null>(async () => {
    if (cacheKey === null) return null;
    const result = await api.GET("/artifacts/{cache_key}/content", {
      params: { path: { cache_key: cacheKey } },
      parseAs: "json",
    });
    return unwrap(result) as unknown as T;
  }, cacheKey ?? "");
}

export function TaskStatus({ task }: { task: TaskOut }) {
  return (
    <span className="flex items-center gap-1">
      <StatusChip status={task.status} />
      {task.error_code ? (
        <code className="text-xs text-danger">{task.error_code}</code>
      ) : null}
    </span>
  );
}

export function Tasks({
  run,
  stage,
  empty,
  children,
}: {
  run: RunOut;
  stage: Stage;
  empty: string;
  children: (tasks: TaskOut[]) => React.ReactNode;
}) {
  const tasks = useStageTasks(run, stage);
  if (tasks.error) return <ErrorNote error={tasks.error} />;
  if (tasks.data === undefined) return <Loading />;
  if (tasks.data.length === 0) return <Empty>{empty}</Empty>;
  return <>{children(tasks.data)}</>;
}
