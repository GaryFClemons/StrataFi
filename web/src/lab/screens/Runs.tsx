import { Button } from "@heroui/react";
import { Suspense, lazy, useState } from "react";
import type { components } from "../api";
import { unwrap, useLoad } from "../api/load";
import { useLiveRun, type RunOut } from "../components/live";
import {
  DataTable,
  Empty,
  ErrorNote,
  Loading,
  Section,
  StatusChip,
  inputClass,
} from "../components/ui";
import { Link, useLab, useNavigate } from "../context";
import {
  formatDuration,
  formatTime,
  isActive,
  progress,
  shortHash,
} from "../format";
import { STAGES } from "../spec/editing";

type RunEntryOut = components["schemas"]["RunEntryOut"];

const RunPanels = lazy(() =>
  import("./Panels").then((module) => ({ default: module.RunPanels })),
);
type TaskOut = components["schemas"]["TaskOut"];

export function RunProgress({ run }: { run: RunOut }) {
  const { finished, total } = progress(run.counts.tasks);
  return (
    <div className="grid gap-1" data-testid="run-progress">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <StatusChip status={run.status} />
        <span>
          {finished} of {total} tasks
          {run.counts.tasks?.cached
            ? `, ${run.counts.tasks.cached} from the cache`
            : ""}
        </span>
        {run.error_code ? (
          <code className="text-danger">{run.error_code}</code>
        ) : null}
      </div>
      <progress
        aria-label="Run progress"
        className="h-1.5 w-full accent-accent"
        max={100}
        value={
          total ? (finished / total) * 100 : isActive(run.status) ? 0 : 100
        }
      />
    </div>
  );
}

export function Runs() {
  const { api } = useLab();
  const [cursor, setCursor] = useState<string | null>(null);
  const runs = useLoad(
    async () =>
      unwrap(
        await api.GET("/runs", {
          params: { query: { limit: 25, ...(cursor ? { cursor } : {}) } },
        }),
      ),
    `runs:${cursor ?? ""}`,
  );
  return (
    <Section
      title="Runs"
      description="Every pipeline run, newest first."
      actions={
        <Button size="sm" variant="secondary" onPress={runs.reload}>
          Refresh
        </Button>
      }
    >
      <ErrorNote error={runs.error} />
      {runs.data === undefined ? (
        <Loading />
      ) : runs.data.items.length === 0 ? (
        <Empty>No runs yet: run a step from Pipelines.</Empty>
      ) : (
        <>
          <DataTable<RunOut>
            label="Runs"
            rows={runs.data.items}
            rowKey={(run) => run.id}
            columns={[
              {
                header: "Run",
                cell: (run) => (
                  <Link
                    to={{ name: "run", runId: run.id }}
                    className="text-link"
                  >
                    <code>{shortHash(run.id, 8)}</code>
                  </Link>
                ),
              },
              {
                header: "Status",
                cell: (run) => <StatusChip status={run.status} />,
              },
              { header: "Until", cell: (run) => run.until ?? "index" },
              {
                header: "Tasks",
                cell: (run) => {
                  const { finished, total } = progress(run.counts.tasks);
                  return `${finished}/${total}`;
                },
              },
              { header: "Mode", cell: (run) => run.mode },
              { header: "Started", cell: (run) => formatTime(run.created_at) },
            ]}
          />
          {runs.data.next_cursor ? (
            <Button
              size="sm"
              variant="tertiary"
              onPress={() => setCursor(runs.data?.next_cursor ?? null)}
            >
              Older runs
            </Button>
          ) : null}
        </>
      )}
    </Section>
  );
}

function StageMatrix({ run }: { run: RunOut }) {
  const statuses = [
    "succeeded",
    "cached",
    "running",
    "queued",
    "failed",
    "cancelled",
  ];
  const byStage = run.counts.stages ?? {};
  const stages = STAGES.filter((stage) => byStage[stage]);
  if (stages.length === 0) return <Empty>No tasks yet.</Empty>;
  return (
    <DataTable<string>
      label="Tasks by stage"
      rows={stages}
      rowKey={(stage) => stage}
      columns={[
        { header: "Stage", cell: (stage) => stage },
        ...statuses.map((status) => ({
          header: status,
          cell: (stage: string) => byStage[stage]?.[status] ?? "",
        })),
      ]}
    />
  );
}

export function RunDetail({ runId }: { runId: string }) {
  const { api, host } = useLab();
  const navigate = useNavigate();
  const live = useLiveRun(runId);
  const run = live.data;
  const finished = run !== undefined && !isActive(run.status);
  const entries = useLoad(
    async () =>
      unwrap(
        await api.GET("/runs/{run_id}/entries", {
          params: { path: { run_id: runId }, query: { limit: 200 } },
        }),
      ),
    `${runId}:${run?.status ?? ""}`,
  );
  const failed = useLoad(
    async () =>
      unwrap(
        await api.GET("/runs/{run_id}/tasks", {
          params: {
            path: { run_id: runId },
            query: { status: "failed", limit: 50 },
          },
        }),
      ),
    `${runId}:${run?.status ?? ""}:failed`,
  );
  const manifest = useLoad(async () => {
    if (!finished) return null;
    return unwrap(
      await api.GET("/runs/{run_id}/manifest", {
        params: { path: { run_id: runId } },
      }),
    );
  }, `${runId}:${finished}`);
  const [fromStage, setFromStage] = useState("parse");
  const [error, setError] = useState<unknown>(null);

  const cancel = async () => {
    const confirmed = await host.confirm({
      title: "Cancel this run?",
      body: "Waiting tasks stop now; running ones at their next checkpoint. Finished work stays cached.",
      confirmLabel: "Cancel run",
      danger: true,
    });
    if (!confirmed) return;
    try {
      unwrap(
        await api.POST("/runs/{run_id}/cancel", {
          params: { path: { run_id: runId } },
        }),
      );
      live.reload();
    } catch (failure) {
      setError(failure);
    }
  };

  const rerun = async () => {
    try {
      const created = unwrap(
        await api.POST("/runs/{run_id}/rerun", {
          params: { path: { run_id: runId } },
          body: {
            from_stage:
              fromStage as components["schemas"]["RerunIn"]["from_stage"],
          },
        }),
      );
      navigate({ name: "run", runId: created.id });
    } catch (failure) {
      setError(failure);
    }
  };

  if (run === undefined) {
    return live.error ? <ErrorNote error={live.error} /> : <Loading />;
  }
  const document = manifest.data;
  return (
    <div className="grid gap-4">
      <p className="text-sm">
        <Link to={{ name: "runs" }} className="text-link">
          Runs
        </Link>{" "}
        / <code>{shortHash(run.id, 8)}</code>
      </p>
      <Section
        title="Run"
        description={`${run.mode} run to ${run.until ?? "index"}${
          run.rerun_from ? `, re-run from ${run.rerun_from}` : ""
        }; started ${formatTime(run.created_at)}`}
        actions={
          isActive(run.status) ? (
            <Button size="sm" variant="danger-soft" onPress={cancel}>
              Cancel
            </Button>
          ) : (
            <div className="flex items-center gap-2">
              <select
                aria-label="Re-run from stage"
                className={inputClass}
                value={fromStage}
                onChange={(event) => setFromStage(event.target.value)}
              >
                {STAGES.map((stage) => (
                  <option key={stage} value={stage}>
                    {stage}
                  </option>
                ))}
              </select>
              <Button size="sm" variant="secondary" onPress={rerun}>
                Re-run from stage
              </Button>
            </div>
          )
        }
      >
        <RunProgress run={run} />
        <ErrorNote error={error} />
        <div className="mt-3">
          <StageMatrix run={run} />
        </div>
      </Section>
      {failed.data && failed.data.items.length > 0 ? (
        <Section title="Errors" description="Codes only: never document text.">
          <DataTable<TaskOut>
            label="Failed tasks"
            rows={failed.data.items}
            rowKey={(task) => task.id}
            columns={[
              { header: "Document", cell: (task) => task.entry ?? "corpus" },
              { header: "Stage", cell: (task) => task.stage },
              {
                header: "Code",
                cell: (task) => <code>{task.error_code}</code>,
              },
              { header: "Attempts", cell: (task) => task.attempts },
            ]}
          />
        </Section>
      ) : null}
      <Section title="Documents" description="Where each document stopped.">
        {entries.data === undefined ? (
          <Loading />
        ) : (
          <DataTable<RunEntryOut>
            label="Run documents"
            rows={entries.data.items}
            rowKey={(entry) => String(entry.ordinal)}
            columns={[
              { header: "Document", cell: (entry) => entry.name },
              {
                header: "State",
                cell: (entry) => <StatusChip status={entry.state} />,
              },
              { header: "Reason", cell: (entry) => entry.stop_code ?? "" },
              {
                header: "Duplicate",
                cell: (entry) => (entry.duplicate ? "yes" : ""),
              },
            ]}
          />
        )}
      </Section>
      {document ? (
        <Section
          title="Manifest"
          description={`Fingerprint ${shortHash(run.manifest_hash, 16)}`}
        >
          <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-sm">
            <dt className="text-muted">Pipeline</dt>
            <dd>
              {String(document.configuration.pipeline_ref?.name ?? "")} v
              {String(document.configuration.pipeline_ref?.version ?? "")} (
              <code>{shortHash(document.configuration.pipeline_hash)}</code>)
            </dd>
            <dt className="text-muted">Corpus</dt>
            <dd>
              v{document.inputs.corpus?.version},{" "}
              {document.inputs.documents?.length ?? 0} documents
            </dd>
            {(document.models ?? []).map((model) => (
              <ModelRow key={model.role} model={model} />
            ))}
            <dt className="text-muted">Environment</dt>
            <dd>
              Stratafi {document.environment.stratafi_version}
              {document.environment.commit
                ? ` (${document.environment.commit})`
                : ""}
              , Python {document.environment.python}
            </dd>
            <dt className="text-muted">Hardware</dt>
            <dd>
              {document.hardware.cpu_model ?? "CPU"} ×{" "}
              {document.hardware.cpu_cores ?? "?"},{" "}
              {document.hardware.gpus?.length ?? 0} GPU(s)
            </dd>
            <dt className="text-muted">Duration</dt>
            <dd>{formatDuration(document.execution.duration_ms)}</dd>
            <dt className="text-muted">Egress</dt>
            <dd>
              {(document.egress ?? []).length === 0
                ? "none"
                : (document.egress ?? [])
                    .map(
                      (item) =>
                        `${item.provider}: ${item.documents?.length ?? 0} documents`,
                    )
                    .join("; ")}
            </dd>
          </dl>
        </Section>
      ) : null}
      <Suspense fallback={<Loading label="Loading the run's map" />}>
        <RunPanels runId={runId} />
      </Suspense>
    </div>
  );
}

function ModelRow({
  model,
}: {
  model: components["schemas"]["ManifestModel"];
}) {
  return (
    <>
      <dt className="text-muted">Model ({model.role})</dt>
      <dd>
        {model.alias} → {model.served_model ?? "?"}, {model.dimensions ?? "?"}{" "}
        dimensions, revision <code>{model.revision}</code>
      </dd>
    </>
  );
}
