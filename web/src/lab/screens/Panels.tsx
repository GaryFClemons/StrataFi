// The tracking panels on existing screens, loaded on first use so the entry
// bundle stays inside its budget: an eval run's annotation, latency
// profile and map; a pipeline run's annotation and map; an experiment's
// annotation; and a query's trace.

import { useState } from "react";
import { unwrap, useLoad } from "../api/load";
import { Annotations } from "../components/Annotations";
import { Chart } from "../components/Chart";
import { PipelineMap } from "../components/PipelineMap";
import { TraceView } from "../components/TraceView";
import {
  DataTable,
  ErrorNote,
  Loading,
  Section,
  inputClass,
} from "../components/ui";
import { Link, useLab } from "../context";
import { STEP_NAMES } from "../compare";
import {
  PROFILE_STEPS,
  histogramOption,
  stepBoxOption,
  type StepProfile,
} from "../traces";

function LatencyProfile({ evalRunId }: { evalRunId: string }) {
  const { api } = useLab();
  const profile = useLoad(
    async () =>
      unwrap(
        await api.GET("/eval-runs/{eval_run_id}/latency", {
          params: { path: { eval_run_id: evalRunId } },
        }),
      ),
    evalRunId,
  );
  const [step, setStep] = useState("total_ms");
  const data = profile.data;
  return (
    <Section
      title="Latency"
      description="Each step's distribution over the run's queries, and its slowest queries."
    >
      {data === undefined ? (
        profile.error ? (
          <ErrorNote error={profile.error} />
        ) : (
          <Loading />
        )
      ) : (
        <div className="grid gap-4 lg:grid-cols-2">
          <Chart
            option={stepBoxOption(
              [data.steps as unknown as Record<string, StepProfile>],
              ["This run"],
            )}
            label="Each step's latency: p25 to p75 with the median, the minimum to p95"
            height={280}
            testId="latency-boxes"
            renderer="svg"
          />
          <div className="grid gap-2">
            <label className="flex items-center gap-2 text-sm font-medium">
              Histogram of
              <select
                className={inputClass}
                value={step}
                onChange={(event) => setStep(event.target.value)}
              >
                {PROFILE_STEPS.filter((name) => name in data.steps).map(
                  (name) => (
                    <option key={name} value={name}>
                      {STEP_NAMES[name] ?? "Total"}
                    </option>
                  ),
                )}
              </select>
            </label>
            {data.steps[step] ? (
              <Chart
                option={histogramOption(data.steps[step].histogram, step)}
                label="How many queries took how long"
                height={240}
                testId="latency-histogram"
                renderer="svg"
              />
            ) : null}
          </div>
          <div className="lg:col-span-2">
            <DataTable
              label="Slowest queries"
              rows={data.slowest}
              rowKey={(row) => row.item_id}
              columns={[
                {
                  header: "Query",
                  cell: (row) => (
                    <Link
                      to={{
                        name: "eval-query",
                        evalRunId,
                        itemId: row.item_id,
                      }}
                      className="text-link"
                    >
                      <code>{row.item_id}</code>
                    </Link>
                  ),
                },
                { header: "Total ms", cell: (row) => row.total_ms.toFixed(1) },
                ...["embed_ms", "keyword_ms", "dense_ms", "rerank_ms"].map(
                  (name) => ({
                    header: STEP_NAMES[name] ?? name,
                    cell: (row: (typeof data.slowest)[number]) =>
                      typeof row.steps[name] === "number"
                        ? row.steps[name].toFixed(1)
                        : "–",
                  }),
                ),
              ]}
            />
          </div>
        </div>
      )}
    </Section>
  );
}

function EvalRunMap({ evalRunId }: { evalRunId: string }) {
  const { api } = useLab();
  const map = useLoad(
    async () =>
      unwrap(
        await api.GET("/eval-runs/{eval_run_id}/map", {
          params: { path: { eval_run_id: evalRunId } },
        }),
      ),
    evalRunId,
  );
  return (
    <Section
      title="Pipeline map"
      description="This eval run's RAG pipeline: its ingest stages with the pipeline run's numbers, its query steps with their latency, and every datastore and service they touch."
    >
      {map.data ? (
        <PipelineMap
          map={map.data}
          mermaid={map.data.mermaid}
          label="This eval run's pipeline map"
        />
      ) : map.error ? (
        <ErrorNote error={map.error} />
      ) : (
        <Loading />
      )}
    </Section>
  );
}

/** An eval run's annotation, and once it has finished, its latency and map. */
export function EvalRunPanels({
  evalRunId,
  finished,
}: {
  evalRunId: string;
  finished: boolean;
}) {
  return (
    <>
      <Annotations kind="eval_run" id={evalRunId} />
      {finished ? (
        <>
          <LatencyProfile evalRunId={evalRunId} />
          <EvalRunMap evalRunId={evalRunId} />
        </>
      ) : null}
    </>
  );
}

/** A pipeline run's annotation and its map with each stage's numbers. */
export function RunPanels({ runId }: { runId: string }) {
  const { api } = useLab();
  const map = useLoad(
    async () =>
      unwrap(
        await api.GET("/runs/{run_id}/map", {
          params: { path: { run_id: runId } },
        }),
      ),
    runId,
  );
  return (
    <>
      <Annotations kind="run" id={runId} />
      <Section
        title="Pipeline map"
        description="This run's stages with their tasks, cache hits, time and bytes, and the datastores and services they touch."
      >
        {map.data ? (
          <PipelineMap
            map={map.data}
            mermaid={map.data.mermaid}
            label="This run's pipeline map"
          />
        ) : map.error ? (
          <ErrorNote error={map.error} />
        ) : (
          <Loading />
        )}
      </Section>
    </>
  );
}

export function ExperimentPanels({ experimentId }: { experimentId: string }) {
  return <Annotations kind="experiment" id={experimentId} />;
}

/** A query's trace: the search's steps on a timeline, each with its lists. */
export function QueryTrace({
  evalRunId,
  itemId,
}: {
  evalRunId: string;
  itemId: string;
}) {
  const { api } = useLab();
  const trace = useLoad(
    async () =>
      unwrap(
        await api.GET("/eval-runs/{eval_run_id}/queries/{item_id}/trace", {
          params: { path: { eval_run_id: evalRunId, item_id: itemId } },
        }),
      ),
    `${evalRunId}:${itemId}`,
  );
  return (
    <Section
      title="Trace"
      description="Retrieval's steps for this query, each where the one before ended, with each list and where the evidence is in it."
    >
      {trace.data ? (
        <TraceView trace={trace.data} />
      ) : trace.error ? (
        <ErrorNote error={trace.error} />
      ) : (
        <Loading />
      )}
    </Section>
  );
}
