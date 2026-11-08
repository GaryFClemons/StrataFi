// The Chunk step. The document's text with
// chunk boundaries (alternating bands, overlaps hatched); the chunk list with
// tokens, heading path and pages; a token histogram with the embedding
// model's input limit drawn in; warnings (over the limit, tiny chunks, split
// tables); a second chunker side by side on the same document; and
// enrichment samples when an enricher added context.

import { Button } from "@heroui/react";
import { useMemo, useState } from "react";
import type { components } from "../../api";
import { unwrap, useLoad } from "../../api/load";
import { Chart } from "../../components/Chart";
import { useLiveRun, type RunOut } from "../../components/live";
import {
  DataTable,
  Empty,
  ErrorNote,
  Loading,
  StatusChip,
  inputClass,
} from "../../components/ui";
import { useLab } from "../../context";
import {
  chunkWarnings,
  coverageBands,
  histogram,
  type ChunkWarning,
} from "../../inspect";
import type { ParsedDocument } from "../../parse";
import { getAt, setAt } from "../../spec/editing";
import { defaultsFromSchema, type JsonSchema } from "../../spec/schemaForm";
import type { Workbench } from "./Parse";
import {
  Tasks,
  TaskStatus,
  summaryOf,
  useArtifactJson,
  useStageTasks,
  value,
  type TaskOut,
} from "./shared";

type ChunkOut = components["schemas"]["ChunkOut"];
type PipelineSpec = components["schemas"]["PipelineSpec"];

function useChunks(cacheKey: string | null) {
  const { api } = useLab();
  return useLoad<ChunkOut[] | null>(async () => {
    if (!cacheKey) return null;
    return unwrap(
      await api.GET("/artifacts/{cache_key}/chunks", {
        params: { path: { cache_key: cacheKey }, query: { limit: 200 } },
      }),
    ).items;
  }, `chunks:${cacheKey}`);
}

function tokensOf(chunk: ChunkOut): number {
  return Object.values(chunk.tokens)[0] ?? 0;
}

/** The embedding model's input limit, as the spec's embedder declares it. */
function specLimit(workbench: Workbench): number {
  const declared = getAt(
    workbench.document,
    "stages.embed.params.max_input_tokens",
  );
  return typeof declared === "number" && declared > 0 ? declared : 512;
}

function TextBands({ text, chunks }: { text: string; chunks: ChunkOut[] }) {
  return (
    <div
      aria-label="Text with chunk boundaries"
      className="max-h-96 overflow-auto whitespace-pre-wrap break-words rounded-md border border-separator p-3 font-mono text-xs"
    >
      {coverageBands(text.length, chunks).map((band) => (
        <span
          key={band.start}
          data-overlap={band.overlap ? "true" : undefined}
          className={
            band.chunk === null
              ? ""
              : band.chunk % 2 === 0
                ? "bg-accent-soft"
                : "bg-default"
          }
          style={
            band.overlap
              ? {
                  backgroundImage:
                    "repeating-linear-gradient(45deg, rgba(127,127,127,0.35) 0 2px, transparent 2px 6px)",
                }
              : undefined
          }
          title={
            band.chunk === null
              ? undefined
              : band.overlap
                ? "Overlap"
                : `Chunk ${band.chunk + 1}`
          }
        >
          {text.slice(band.start, band.end)}
        </span>
      ))}
    </div>
  );
}

function TokenHistogram({
  chunks,
  limit,
}: {
  chunks: ChunkOut[];
  limit: number;
}) {
  const option = useMemo(() => {
    const bins = histogram(chunks.map(tokensOf), 20, limit);
    return {
      grid: { left: 40, right: 16, top: 24, bottom: 32 },
      tooltip: { trigger: "axis" },
      xAxis: {
        type: "category",
        name: "tokens",
        data: bins.map((bin) => `${bin.from}`),
      },
      yAxis: { type: "value", name: "chunks", minInterval: 1 },
      series: [
        {
          type: "bar",
          data: bins.map((bin) => bin.count),
          itemStyle: { color: "#1f77b4" },
          markLine: {
            symbol: "none",
            label: { formatter: `limit ${limit}` },
            lineStyle: { color: "#d62728", type: "dashed" },
            data: [
              {
                xAxis: `${bins.find((bin) => limit < bin.to)?.from ?? bins.at(-1)!.from}`,
              },
            ],
          },
        },
      ],
    };
  }, [chunks, limit]);
  return <Chart option={option} label="Token histogram with the input limit" />;
}

function Warnings({ warnings }: { warnings: ChunkWarning[] }) {
  if (warnings.length === 0)
    return (
      <p className="text-xs text-muted" aria-label="Chunk warnings">
        No warnings.
      </p>
    );
  return (
    <ul aria-label="Chunk warnings" className="grid gap-1 text-xs">
      {warnings.map((warning, index) => (
        <li key={index} className="flex items-center gap-2">
          <StatusChip status="flag" label={warning.kind.replace(/_/g, " ")} />
          {warning.chunk !== null ? `chunk ${warning.chunk + 1}: ` : ""}
          {warning.detail}
        </li>
      ))}
    </ul>
  );
}

function ChunkList({ chunks, label }: { chunks: ChunkOut[]; label: string }) {
  return (
    <DataTable<ChunkOut>
      label={label}
      rows={chunks}
      rowKey={(chunk) => chunk.chunk_id}
      columns={[
        { header: "#", cell: (chunk) => chunk.ordinal + 1 },
        {
          header: "Tokens",
          cell: (chunk) =>
            Object.entries(chunk.tokens)
              .map(([estimator, count]) => `${count} (${estimator})`)
              .join(", "),
        },
        {
          header: "Heading path",
          cell: (chunk) => chunk.heading_path.join(" › ") || "–",
        },
        {
          header: "Pages",
          cell: (chunk) =>
            chunk.page_start
              ? `${chunk.page_start}${chunk.page_end !== chunk.page_start ? `–${chunk.page_end}` : ""}`
              : "–",
        },
        {
          header: "Text",
          cell: (chunk) => (
            <span className="line-clamp-2 whitespace-pre-wrap">
              {chunk.text}
            </span>
          ),
        },
      ]}
    />
  );
}

function EnrichmentSamples({ chunks }: { chunks: ChunkOut[] }) {
  const enriched = chunks.filter((chunk) => chunk.context).slice(0, 5);
  if (enriched.length === 0) return null;
  return (
    <section aria-label="Enrichment samples" className="grid gap-2">
      <h3 className="text-sm font-semibold">Context added by enrichment</h3>
      {enriched.map((chunk) => (
        <div
          key={chunk.chunk_id}
          className="grid gap-1 rounded-md border border-separator p-2 text-xs"
        >
          <span className="font-medium">{chunk.context}</span>
          <span className="line-clamp-2 whitespace-pre-wrap text-muted">
            {chunk.text}
          </span>
        </div>
      ))}
    </section>
  );
}

function ChunkView({
  chunksKey,
  cleanedKey,
  limit,
}: {
  chunksKey: string;
  cleanedKey: string | null;
  limit: number;
}) {
  const chunks = useChunks(chunksKey);
  const cleaned = useArtifactJson<ParsedDocument>(cleanedKey);
  const [tiny, setTiny] = useState(32);
  if (chunks.error) return <ErrorNote error={chunks.error} />;
  if (!chunks.data) return <Loading />;
  const elements = cleaned.data?.elements ?? [];
  const warnings = chunkWarnings(chunks.data, elements, limit, tiny);
  const parents = new Set(
    chunks.data
      .map(
        (chunk) => (chunk.metadata as { parent?: { id?: string } }).parent?.id,
      )
      .filter(Boolean),
  );
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-end gap-3 text-xs">
        <span>
          {chunks.data.length} chunks
          {parents.size ? ` · ${parents.size} parent sections` : ""}
        </span>
        <label className="flex items-center gap-1">
          Tiny under
          <input
            type="number"
            className={`${inputClass} w-20`}
            value={tiny}
            min={0}
            onChange={(event) => setTiny(Number(event.target.value) || 0)}
          />
          tokens
        </label>
      </div>
      <TokenHistogram chunks={chunks.data} limit={limit} />
      <Warnings warnings={warnings} />
      {cleaned.data ? (
        <TextBands text={cleaned.data.text} chunks={chunks.data} />
      ) : null}
      <ChunkList chunks={chunks.data} label="Chunks" />
      <EnrichmentSamples chunks={chunks.data} />
    </div>
  );
}

function CompareChunker({ runId, limit }: { runId: string; limit: number }) {
  const live = useLiveRun(runId);
  const { api } = useLab();
  const done = live.data?.status === "succeeded";
  const tasks = useLoad(async () => {
    if (!done) return null;
    const list = async (stage: "chunk" | "clean") =>
      unwrap(
        await api.GET("/runs/{run_id}/tasks", {
          params: { path: { run_id: runId }, query: { stage } },
        }),
      ).items[0] ?? null;
    return { chunk: await list("chunk"), clean: await list("clean") };
  }, `${runId}:compare-chunk:${done}`);
  if (live.error) return <ErrorNote error={live.error} />;
  if (!tasks.data?.chunk) return <Loading label="Chunking" />;
  return (
    <div className="grid content-start gap-2" data-testid="compare-chunker">
      <p className="text-xs font-semibold">{tasks.data.chunk.component_id}</p>
      <ChunkView
        chunksKey={tasks.data.chunk.cache_key}
        cleanedKey={tasks.data.clean?.cache_key ?? null}
        limit={limit}
      />
    </div>
  );
}

function CompareChunkers({
  run,
  task,
  workbench,
  limit,
}: {
  run: RunOut;
  task: TaskOut;
  workbench: Workbench;
  limit: number;
}) {
  const { api } = useLab();
  const components = useLoad(
    async () =>
      unwrap(
        await api.GET("/components", { params: { query: { stage: "chunk" } } }),
      ).items,
    "chunk-components",
  );
  const [chosen, setChosen] = useState("");
  const [compared, setCompared] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);
  const others = (components.data ?? []).filter(
    (item) => item.id !== task.component_id && item.available,
  );
  const pick = chosen || others[0]?.id || "";
  const compare = async () => {
    const component = others.find((item) => item.id === pick);
    if (!workbench.document || !task.entry || !component) return;
    setError(null);
    try {
      let variant = setAt(
        workbench.document,
        "stages.chunk.component",
        component.id,
      );
      variant = setAt(
        variant,
        "stages.chunk.params",
        defaultsFromSchema(component.params_schema as JsonSchema),
      );
      const saved = unwrap(
        await api.POST("/pipeline-specs", {
          body: {
            ...variant,
            name: `${workbench.specName}-compare-${component.id.split("/")[1]}`.slice(
              0,
              64,
            ),
          } as PipelineSpec,
        }),
      );
      const started = unwrap(
        await api.POST("/runs", {
          body: {
            pipeline_spec_id: saved.id,
            corpus_version_id: run.corpus_version_id,
            until: "enrich",
            sample: { entries: [task.entry] },
            mode: "interactive",
          },
        }),
      );
      setCompared(started.id);
    } catch (failure) {
      setError(failure);
    }
  };
  return (
    <section
      aria-label="Compare chunkers"
      className="grid gap-2 rounded-md border border-separator p-3"
    >
      <div className="flex flex-wrap items-end gap-2 text-xs">
        <label className="grid gap-1 font-medium">
          Compare with
          <select
            className={inputClass}
            value={pick}
            onChange={(event) => setChosen(event.target.value)}
          >
            {others.map((item) => (
              <option key={item.id} value={item.id}>
                {item.title}
              </option>
            ))}
          </select>
        </label>
        <Button
          size="sm"
          variant="secondary"
          isDisabled={!pick}
          onPress={() => void compare()}
        >
          Compare chunkers
        </Button>
      </div>
      <ErrorNote error={error ?? components.error} />
      {compared ? (
        <CompareChunker key={compared} runId={compared} limit={limit} />
      ) : null}
    </section>
  );
}

export function ChunkInspector({
  run,
  workbench,
}: {
  run: RunOut;
  workbench: Workbench;
}) {
  const [chosen, setChosen] = useState<string | null>(null);
  const [limit, setLimit] = useState<number>(() => specLimit(workbench));
  const cleaned = useStageTasks(run, "clean");
  return (
    <Tasks run={run} stage="chunk" empty="No documents chunked yet.">
      {(tasks) => {
        const done = tasks.filter(
          (task) => summaryOf(task).chunk_count !== undefined,
        );
        const shown =
          tasks.find((task) => task.cache_key === chosen) ?? done[0] ?? null;
        const cleanedKey =
          cleaned.data?.find((task) => shown && task.ordinal === shown.ordinal)
            ?.cache_key ?? null;
        return (
          <div className="grid gap-3">
            <label className="flex items-center gap-2 text-xs font-medium">
              Embedding model&apos;s input limit
              <input
                type="number"
                className={`${inputClass} w-24`}
                value={limit}
                min={1}
                onChange={(event) => setLimit(Number(event.target.value) || 1)}
              />
              tokens
            </label>
            <DataTable<TaskOut>
              label="Chunked documents"
              rows={tasks}
              rowKey={(task) => task.cache_key}
              selected={shown?.cache_key ?? null}
              onSelect={(task) => setChosen(task.cache_key)}
              columns={[
                { header: "Document", cell: (task) => task.entry ?? "–" },
                { header: "Chunker", cell: (task) => task.component_id },
                {
                  header: "Chunks",
                  cell: (task) => value(summaryOf(task), "chunk_count"),
                },
                {
                  header: "Tokens",
                  cell: (task) =>
                    `${value(summaryOf(task), "tokens_min")}–${value(summaryOf(task), "tokens_max")}`,
                },
                {
                  header: "Status",
                  cell: (task) => <TaskStatus task={task} />,
                },
              ]}
            />
            {shown && summaryOf(shown).chunk_count !== undefined ? (
              <>
                <ChunkView
                  chunksKey={shown.cache_key}
                  cleanedKey={cleanedKey}
                  limit={limit}
                />
                <CompareChunkers
                  run={run}
                  task={shown}
                  workbench={workbench}
                  limit={limit}
                />
              </>
            ) : (
              <Empty>Choose a chunked document.</Empty>
            )}
          </div>
        );
      }}
    </Tasks>
  );
}
