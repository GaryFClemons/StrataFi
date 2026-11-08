// The Embed step. A model catalog: where each
// model runs (gateway, in process, hosted), dimensions, input limit, license,
// and the tokens and time this corpus would take; results: throughput,
// truncated inputs, stored size and dtype; a 2-D projection (PCA, or UMAP
// with its extra) coloured by document or type, with a nearest-neighbour
// explorer; and the indexes with their ANN recall.

import { Button } from "@heroui/react";
import { useMemo, useState } from "react";
import type { components } from "../../api";
import { unwrap, useLoad } from "../../api/load";
import { Chart } from "../../components/Chart";
import type { RunOut } from "../../components/live";
import {
  DataTable,
  Empty,
  ErrorNote,
  Loading,
  StatusChip,
  inputClass,
} from "../../components/ui";
import { useLab } from "../../context";
import { formatBytes, formatDuration, shortHash } from "../../format";
import { setAt, type SpecDocument } from "../../spec/editing";
import {
  Tasks,
  TaskStatus,
  summaryOf,
  useStageTasks,
  value,
  type TaskOut,
} from "./shared";

type CatalogModelOut = components["schemas"]["CatalogModelOut"];
type ProjectionOut = components["schemas"]["ProjectionOut"];

// One palette for every chart, checked for colour-blind safety (Okabe–Ito).
export const PALETTE = [
  "#0072B2",
  "#E69F00",
  "#009E73",
  "#CC79A7",
  "#56B4E9",
  "#D55E00",
  "#F0E442",
  "#000000",
];

/** Use a catalog model in the spec's embed slot. */
export function withModel(
  document: SpecDocument,
  model: CatalogModelOut,
): SpecDocument {
  if (model.component === "embed/gateway")
    return setAt(document, "stages.embed", {
      component: "embed/gateway",
      params: { alias: model.model },
    });
  if (model.component === "embed/sentence-transformers")
    return setAt(document, "stages.embed", {
      component: "embed/sentence-transformers",
      params: { model: model.model, max_input_tokens: model.max_tokens },
    });
  return setAt(document, "stages.embed", {
    component: model.component,
    params: { provider: model.model.split(":")[0] },
  });
}

export function ModelCatalog({
  runId,
  document,
  onChange,
}: {
  runId: string | null;
  document: SpecDocument;
  onChange: (document: SpecDocument) => void;
}) {
  const { api } = useLab();
  const catalog = useLoad(
    async () => unwrap(await api.GET("/models/catalog")).models,
    "model-catalog",
  );
  const chunks = useLoad(
    async () => {
      if (!runId) return null;
      return unwrap(
        await api.GET("/runs/{run_id}/tasks", {
          params: {
            path: { run_id: runId },
            query: { stage: "chunk", limit: 200 },
          },
        }),
      ).items;
    },
    `catalog-chunks:${runId ?? ""}`,
  );
  const tokens = (chunks.data ?? []).reduce((total, task) => {
    const summary = summaryOf(task);
    const count = Number(summary.chunk_count ?? 0);
    const average =
      (Number(summary.tokens_min ?? 0) + Number(summary.tokens_max ?? 0)) / 2;
    return total + count * average;
  }, 0);
  if (catalog.error) return <ErrorNote error={catalog.error} />;
  if (!catalog.data) return <Loading />;
  return (
    <section aria-label="Model catalog" className="grid gap-2">
      <h3 className="text-sm font-semibold">Model catalog</h3>
      {tokens ? (
        <p className="text-xs text-muted">
          This corpus: about {Math.round(tokens).toLocaleString()} tokens to
          embed.
        </p>
      ) : null}
      <DataTable<CatalogModelOut>
        label="Embedding models"
        rows={catalog.data}
        rowKey={(model) => `${model.where}:${model.model}`}
        columns={[
          { header: "Model", cell: (model) => model.title },
          { header: "Runs", cell: (model) => model.where.replace("_", " ") },
          { header: "Dimensions", cell: (model) => model.dims ?? "–" },
          {
            header: "Matryoshka",
            cell: (model) => (model.matryoshka ?? []).join(", ") || "–",
          },
          { header: "Input limit", cell: (model) => model.max_tokens ?? "–" },
          { header: "License", cell: (model) => model.license },
          {
            header: "",
            cell: (model) =>
              model.available ? (
                <Button
                  size="sm"
                  variant="tertiary"
                  aria-label={`Use ${model.title}`}
                  onPress={() => onChange(withModel(document, model))}
                >
                  Use
                </Button>
              ) : (
                <span className="text-xs text-muted">{model.reason}</span>
              ),
          },
        ]}
      />
    </section>
  );
}

function Projection({ run }: { run: RunOut }) {
  const { api } = useLab();
  const [method, setMethod] = useState<"pca" | "umap">("pca");
  const [colour, setColour] = useState<"entry" | "media_type">("entry");
  const [chosen, setChosen] = useState<string | null>(null);
  const projection = useLoad<ProjectionOut>(
    async () =>
      unwrap(
        await api.GET("/runs/{run_id}/projection", {
          params: { path: { run_id: run.id }, query: { method, limit: 5000 } },
        }),
      ),
    `projection:${run.id}:${method}`,
  );
  const neighbours = useLoad(
    async () => {
      if (!chosen) return null;
      return unwrap(
        await api.GET("/runs/{run_id}/neighbours", {
          params: {
            path: { run_id: run.id },
            query: { chunk_id: chosen, k: 8 },
          },
        }),
      ).neighbours;
    },
    `neighbours:${run.id}:${chosen ?? ""}`,
  );
  const groups = useMemo(() => {
    const points = projection.data?.points ?? [];
    const byGroup = new Map<string, typeof points>();
    for (const point of points) {
      const key = String(point[colour] ?? "unknown");
      byGroup.set(key, [...(byGroup.get(key) ?? []), point]);
    }
    return [...byGroup.entries()];
  }, [projection.data, colour]);
  const option = useMemo(
    () => ({
      grid: { left: 40, right: 16, top: 32, bottom: 32 },
      tooltip: { trigger: "item", formatter: "{a}" },
      legend: { type: "scroll", top: 0 },
      xAxis: { type: "value", scale: true },
      yAxis: { type: "value", scale: true },
      series: groups.map(([name, points], index) => ({
        type: "scatter",
        name,
        symbolSize: 7,
        large: points.length > 2000,
        itemStyle: { color: PALETTE[index % PALETTE.length] },
        data: points.map((point) => [point.x, point.y]),
      })),
    }),
    [groups],
  );
  if (projection.error) return <ErrorNote error={projection.error} />;
  if (!projection.data) return <Loading label="Projecting" />;
  return (
    <section aria-label="Projection" className="grid gap-2">
      <div className="flex flex-wrap items-end gap-3 text-xs">
        <h3 className="text-sm font-semibold">Projection</h3>
        <label className="flex items-center gap-1">
          Method
          <select
            className={inputClass}
            value={method}
            onChange={(event) =>
              setMethod(event.target.value as "pca" | "umap")
            }
          >
            <option value="pca">PCA</option>
            <option value="umap">UMAP</option>
          </select>
        </label>
        <label className="flex items-center gap-1">
          Colour by
          <select
            className={inputClass}
            value={colour}
            onChange={(event) =>
              setColour(event.target.value as "entry" | "media_type")
            }
          >
            <option value="entry">Document</option>
            <option value="media_type">Type</option>
          </select>
        </label>
        <span className="text-muted">
          {projection.data.points.length} of {projection.data.total} vectors
          {projection.data.explained_variance?.length
            ? ` · variance ${projection.data.explained_variance
                .map((share) => `${Math.round(share * 100)}%`)
                .join(" + ")}`
            : ""}
        </span>
      </div>
      <Chart
        option={option}
        label="Projection of the run's vectors"
        height={320}
        onClick={({ seriesIndex, dataIndex }) => {
          const point = groups[seriesIndex]?.[1][dataIndex];
          if (point) setChosen(point.chunk_id);
        }}
      />
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <label className="flex items-center gap-1">
          Explore a chunk
          <select
            className={inputClass}
            value={chosen ?? ""}
            onChange={(event) => setChosen(event.target.value || null)}
          >
            <option value="">Choose…</option>
            {projection.data.points.slice(0, 500).map((point) => (
              <option key={point.chunk_id} value={point.chunk_id}>
                {point.entry ?? point.document_id} ·{" "}
                {shortHash(point.chunk_id, 8)}
              </option>
            ))}
          </select>
        </label>
      </div>
      {chosen ? (
        neighbours.error ? (
          <ErrorNote error={neighbours.error} />
        ) : !neighbours.data ? (
          <Loading />
        ) : (
          <ol aria-label="Nearest neighbours" className="grid gap-1 text-xs">
            {neighbours.data.map((item) => (
              <li
                key={item.chunk_id}
                className="grid gap-1 rounded-md border border-separator p-2"
                data-testid="neighbour"
              >
                <span className="text-muted">
                  {item.entry ?? item.document_id} · cosine{" "}
                  {item.score.toFixed(3)}
                </span>
                <span className="line-clamp-2 whitespace-pre-wrap">
                  {item.text}
                </span>
              </li>
            ))}
          </ol>
        )
      ) : null}
    </section>
  );
}

function throughput(task: TaskOut): string {
  const chunks = Number(summaryOf(task).chunks ?? 0);
  if (!task.duration_ms || !chunks || task.cache_hit) return "–";
  return `${(chunks / (task.duration_ms / 1000)).toFixed(1)}/s`;
}

export function EmbedInspector({ run }: { run: RunOut }) {
  const indexes = useStageTasks(run, "index");
  const dense = (indexes.data ?? []).some(
    (task) =>
      summaryOf(task).index_kind === "dense" &&
      task.component_id === "index/exact",
  );
  return (
    <div className="grid gap-4">
      <Tasks run={run} stage="embed" empty="No documents embedded yet.">
        {(tasks) => (
          <DataTable<TaskOut>
            label="Embeddings"
            rows={tasks}
            rowKey={(task) => task.id}
            columns={[
              { header: "Document", cell: (task) => task.entry ?? "–" },
              {
                header: "Chunks",
                cell: (task) => value(summaryOf(task), "chunks"),
              },
              {
                header: "Dimensions",
                cell: (task) => {
                  const summary = summaryOf(task);
                  return summary.full_dims && summary.full_dims !== summary.dims
                    ? `${value(summary, "dims")} of ${value(summary, "full_dims")}`
                    : value(summary, "dims");
                },
              },
              {
                header: "dtype",
                cell: (task) => value(summaryOf(task), "dtype"),
              },
              {
                header: "Size",
                cell: (task) =>
                  formatBytes(summaryOf(task).bytes as number | undefined),
              },
              {
                header: "Truncated",
                cell: (task) =>
                  Number(summaryOf(task).truncated_inputs ?? 0) > 0 ? (
                    <StatusChip
                      status="flag"
                      label={`${value(summaryOf(task), "truncated_inputs")} over the limit`}
                    />
                  ) : (
                    "0"
                  ),
              },
              { header: "Throughput", cell: throughput },
              {
                header: "Model",
                cell: (task) => value(summaryOf(task), "model"),
              },
              {
                header: "Revision",
                cell: (task) => (
                  <code>
                    {shortHash(summaryOf(task).revision as string, 16)}
                  </code>
                ),
              },
              {
                header: "Time",
                cell: (task) => formatDuration(task.duration_ms),
              },
              { header: "Status", cell: (task) => <TaskStatus task={task} /> },
            ]}
          />
        )}
      </Tasks>
      <Tasks run={run} stage="index" empty="No indexes built yet.">
        {(tasks) => (
          <DataTable<TaskOut>
            label="Indexes"
            rows={tasks}
            rowKey={(task) => task.id}
            columns={[
              { header: "Index", cell: (task) => task.component_id },
              {
                header: "Kind",
                cell: (task) => value(summaryOf(task), "index_kind"),
              },
              {
                header: "Chunks",
                cell: (task) => value(summaryOf(task), "chunk_count"),
              },
              {
                header: "Dimensions",
                cell: (task) => value(summaryOf(task), "dims"),
              },
              {
                header: "ANN recall@10",
                cell: (task) => {
                  const recall = task.metrics?.ann_recall_at_10;
                  return typeof recall === "number"
                    ? recall.toFixed(3)
                    : "exact";
                },
              },
              { header: "Status", cell: (task) => <TaskStatus task={task} /> },
            ]}
          />
        )}
      </Tasks>
      {dense ? (
        <Projection run={run} />
      ) : indexes.data ? (
        <Empty>Projections read the exact dense index.</Empty>
      ) : null}
    </div>
  );
}
