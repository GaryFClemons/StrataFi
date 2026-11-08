// ECharts options for the Compare workspace: up to six eval runs, each in its
// own colour and shape in every chart. Pure functions of the API's numbers,
// tested without a browser. Text uses currentColor so charts follow the
// theme; there's never a second y axis.

import { PALETTE, RAMP } from "./charts";
import { runColor, runSymbol } from "./tracking";

type Option = Record<string, unknown>;
const TEXT = "currentColor";
const MUTED = "#8a8a8a";

function axis(name: string, extra: Option = {}): Option {
  return {
    name,
    nameLocation: "middle",
    nameGap: 30,
    nameTextStyle: { color: TEXT },
    axisLabel: { color: TEXT },
    axisLine: { lineStyle: { color: MUTED } },
    splitLine: { lineStyle: { color: MUTED, opacity: 0.25 } },
    ...extra,
  };
}

function legend(labels: string[]): Option {
  // A plain legend wraps onto more lines rather than paging (grids leave room).
  return { data: labels, textStyle: { color: TEXT }, top: 0, itemGap: 14 };
}

/** Grid space under a legend of ``count`` runs: a line per two, as half-width
 * charts wrap them. */
function legendTop(count: number): number {
  return 24 * Math.ceil(count / 2) + 28;
}

/** Run labels for legends: "1. hybrid", so two runs with one title differ;
 * long titles are cut, the number keeps them apart. */
export function runLabels(runs: { label: string }[], length = 30): string[] {
  return runs.map((run, index) => {
    const label = `${index + 1}. ${run.label}`;
    return label.length > length ? `${label.slice(0, length - 1)}…` : label;
  });
}

export interface Cell {
  mean?: number | null;
  low?: number | null;
  high?: number | null;
}

export interface MetricRow {
  metric: string;
  cells: Cell[];
}

interface RenderApi {
  value(dimension: number): number;
  coord(point: [number, number]): [number, number];
  size(size: [number, number]): [number, number];
  visual(name: string): string;
}

/** A row's whisker and dot, offset within its metric's band by the run's place. */
export function forestMark(runs: number) {
  return (params: { seriesIndex: number }, api: RenderApi): Option => {
    const row = api.value(0);
    const [x1, y] = api.coord([api.value(2), row]);
    const [x2] = api.coord([api.value(3), row]);
    const [xm] = api.coord([api.value(1), row]);
    const band = Math.abs(api.size([0, 1])[1]);
    const offset = (params.seriesIndex - (runs - 1) / 2) * (band / (runs + 1));
    const color = api.visual("color");
    const yy = y + offset;
    return {
      type: "group",
      children: [
        {
          type: "line",
          shape: { x1, y1: yy, x2, y2: yy },
          style: { stroke: color, lineWidth: 2 },
        },
        {
          type: "circle",
          shape: { cx: xm, cy: yy, r: 5 },
          style: { fill: color, stroke: "#ffffff", lineWidth: 2 },
        },
      ],
    };
  };
}

/** Each metric's mean and 95% interval for every run: a dot plot, one band per
 * metric, the runs offset inside it. */
export function forestOption(rows: MetricRow[], labels: string[]): Option {
  const values = rows.flatMap((row) =>
    row.cells.flatMap((cell) =>
      [cell.low, cell.high, cell.mean].filter(
        (value): value is number => typeof value === "number",
      ),
    ),
  );
  const low = values.length ? Math.min(...values) : 0;
  const high = values.length ? Math.max(...values) : 1;
  const pad = (high - low || 0.1) * 0.08;
  return {
    animation: false,
    aria: { enabled: true },
    legend: legend(labels),
    grid: {
      left: 16,
      right: 24,
      top: legendTop(labels.length),
      bottom: 40,
      containLabel: true,
    },
    tooltip: { trigger: "item" },
    xAxis: axis("Mean with its 95% interval", {
      type: "value",
      min: Math.max(0, Math.floor((low - pad) * 20) / 20),
      max: Math.min(1, Math.ceil((high + pad) * 20) / 20),
    }),
    yAxis: axis("", {
      type: "category",
      data: rows.map((row) => row.metric),
      inverse: true,
    }),
    series: labels.map((label, index) => ({
      name: label,
      type: "custom",
      color: runColor(index),
      renderItem: forestMark(labels.length),
      encode: { x: [1, 2, 3], y: 0 },
      data: rows
        .map((row, rowIndex) => {
          const cell = row.cells[index];
          if (typeof cell?.mean !== "number") return null;
          return [
            rowIndex,
            cell.mean,
            cell.low ?? cell.mean,
            cell.high ?? cell.mean,
          ];
        })
        .filter((item) => item !== null),
    })),
  };
}

export interface Curve {
  metric: string;
  ks: string[];
  series: (number | null)[][];
}

/** A metric at every k, one line per run: how fast each pipeline finds the
 * evidence as the list grows. */
export function curvesOption(curve: Curve, labels: string[]): Option {
  return {
    animation: false,
    aria: { enabled: true },
    legend: legend(labels),
    grid: {
      left: 16,
      right: 24,
      top: legendTop(labels.length),
      bottom: 40,
      containLabel: true,
    },
    tooltip: { trigger: "axis" },
    xAxis: axis("k", {
      type: "category",
      data: curve.ks.map((k) => (k === "all" ? "all" : `@${k}`)),
    }),
    yAxis: axis(curve.metric, {
      type: "value",
      min: 0,
      max: 1,
      nameLocation: "end",
      nameGap: 12,
    }),
    series: labels.map((label, index) => ({
      name: label,
      type: "line",
      color: runColor(index),
      symbol: runSymbol(index),
      symbolSize: 8,
      lineStyle: { width: 2 },
      data: curve.series[index] ?? [],
      connectNulls: true,
    })),
  };
}

export const CHECKPOINT_NAMES: Record<string, string> = {
  dense: "Dense",
  keyword: "Keyword",
  fused: "Fused",
  candidates: "Candidates",
  reranked: "Reranked",
  final: "Final",
};

/** The share of queries with a relevant chunk anywhere in each checkpoint's
 * list, one line per run: where each pipeline loses its evidence. */
export function funnelOption(
  funnel: { checkpoints: string[]; hit: (number | null)[][] },
  labels: string[],
): Option {
  return {
    animation: false,
    aria: { enabled: true },
    legend: legend(labels),
    grid: {
      left: 16,
      right: 24,
      top: legendTop(labels.length),
      bottom: 40,
      containLabel: true,
    },
    tooltip: { trigger: "axis" },
    xAxis: axis("List", {
      type: "category",
      data: funnel.checkpoints.map((name) => CHECKPOINT_NAMES[name] ?? name),
    }),
    yAxis: axis("Share with evidence", {
      type: "value",
      min: 0,
      max: 1,
      nameLocation: "end",
      nameGap: 12,
    }),
    series: labels.map((label, index) => ({
      name: label,
      type: "line",
      color: runColor(index),
      symbol: runSymbol(index),
      symbolSize: 8,
      data: funnel.hit[index] ?? [],
      connectNulls: true,
    })),
  };
}

const BUCKETS: Record<string, string> = {
  found: "Found",
  parse: "Lost at parse",
  retrieval: "Lost at retrieval",
  rerank: "Lost at rerank",
};

/** Where each run's queries were found or lost, stacked per run. */
export function lostOption(
  lost: { buckets: string[]; counts: (number | null)[][] },
  labels: string[],
): Option {
  const shown = lost.buckets
    .map((bucket, index) => ({ bucket, index }))
    .filter(({ bucket }) => bucket in BUCKETS);
  const colors = [PALETTE[2], PALETTE[5], PALETTE[1], PALETTE[3]];
  return {
    animation: false,
    aria: { enabled: true },
    color: colors,
    legend: { textStyle: { color: TEXT }, top: 0 },
    grid: { left: 16, right: 24, top: 36, bottom: 40, containLabel: true },
    tooltip: { trigger: "axis" },
    xAxis: axis("Queries", { type: "value" }),
    yAxis: axis("", { type: "category", data: labels, inverse: true }),
    series: shown.map(({ bucket, index }) => ({
      name: BUCKETS[bucket],
      type: "bar",
      stack: "queries",
      barMaxWidth: 28,
      itemStyle: { borderColor: "transparent", borderWidth: 1 },
      data: lost.counts.map((counts) => counts[index] ?? 0),
    })),
  };
}

export const STEP_NAMES: Record<string, string> = {
  keyword_ms: "Keyword",
  embed_ms: "Embed query",
  dense_ms: "Dense",
  fusion_ms: "Fuse and filter",
  rerank_ms: "Rerank",
};

/** Each run's median time per step, stacked: what a query spends where. */
export function latencyOption(
  latency: { steps: string[]; p50: (number | null)[][] },
  labels: string[],
): Option {
  const steps = latency.steps
    .map((step, index) => ({ step, index }))
    .filter(({ step }) => step in STEP_NAMES);
  return {
    animation: false,
    aria: { enabled: true },
    color: PALETTE,
    legend: { textStyle: { color: TEXT }, top: 0 },
    grid: { left: 16, right: 24, top: 36, bottom: 40, containLabel: true },
    tooltip: { trigger: "axis", valueFormatter: (v: number) => `${v} ms` },
    xAxis: axis("Median ms per query", { type: "value" }),
    yAxis: axis("", { type: "category", data: labels, inverse: true }),
    series: steps.map(({ step, index }) => ({
      name: STEP_NAMES[step],
      type: "bar",
      stack: "latency",
      barMaxWidth: 28,
      data: latency.p50.map((values) => values[index] ?? 0),
    })),
  };
}

export interface QualityPoint {
  label: string;
  quality: number | null;
  latency: number | null;
}

/** Quality against p95 latency, a point per run; the runs no other run beats
 * on both are joined as the frontier. */
export function qualityLatencyOption(
  points: QualityPoint[],
  metric: string,
): Option {
  const known = points
    .map((point, index) => ({ ...point, index }))
    .filter(
      (
        point,
      ): point is QualityPoint & {
        index: number;
        quality: number;
        latency: number;
      } =>
        typeof point.quality === "number" && typeof point.latency === "number",
    );
  const frontier = known
    .filter(
      (point) =>
        !known.some(
          (other) =>
            other !== point &&
            other.quality >= point.quality &&
            other.latency <= point.latency &&
            (other.quality > point.quality || other.latency < point.latency),
        ),
    )
    .sort((a, b) => a.latency - b.latency);
  return {
    animation: false,
    aria: { enabled: true },
    legend: legend(points.map((point) => point.label)),
    grid: {
      left: 16,
      right: 24,
      top: legendTop(points.length),
      bottom: 40,
      containLabel: true,
    },
    tooltip: { trigger: "item" },
    xAxis: axis("p95 latency (ms)", { type: "value", scale: true }),
    yAxis: axis(metric, {
      type: "value",
      scale: true,
      nameLocation: "end",
      nameGap: 12,
    }),
    series: [
      ...known.map((point) => ({
        name: point.label,
        type: "scatter",
        color: runColor(point.index),
        symbol: runSymbol(point.index),
        symbolSize: 14,
        itemStyle: { borderColor: "#ffffff", borderWidth: 2 },
        data: [[point.latency, point.quality]],
      })),
      {
        name: "Frontier",
        type: "line",
        color: MUTED,
        symbol: "none",
        lineStyle: { width: 1.5 },
        data: frontier.map((point) => [point.latency, point.quality]),
        z: 1,
      },
    ],
  };
}

export interface ParallelDimension {
  name: string;
  kind: "category" | "value";
  values: unknown[]; // one per run
}

/** Parallel coordinates: the parameters that differ, then the metrics and
 * latency; one line per run, in its colour. */
export function parallelOption(
  dimensions: ParallelDimension[],
  labels: string[],
): Option {
  return {
    animation: false,
    aria: { enabled: true },
    legend: legend(labels),
    parallel: {
      left: 40,
      right: 80,
      top: legendTop(labels.length) + 28,
      bottom: 36,
    },
    tooltip: { trigger: "item" },
    parallelAxis: dimensions.map((dimension, index) => ({
      dim: index,
      name: dimension.name,
      nameTextStyle: { color: TEXT, width: 110, overflow: "truncate" },
      axisLabel: { color: TEXT, width: 110, overflow: "truncate" },
      axisLine: { lineStyle: { color: MUTED } },
      ...(dimension.kind === "category"
        ? {
            type: "category",
            data: [
              ...new Set(dimension.values.map((value) => String(value ?? "–"))),
            ],
          }
        : { type: "value", scale: true }),
    })),
    series: labels.map((label, index) => ({
      name: label,
      type: "parallel",
      color: runColor(index),
      lineStyle: { width: 3, opacity: 0.9 },
      data: [
        dimensions.map((dimension) => {
          const value = dimension.values[index];
          return dimension.kind === "category" ? String(value ?? "–") : value;
        }),
      ],
    })),
  };
}

/** A heatmap of runs (columns) against rows, on the sequential ramp, each cell
 * labelled: slices by file type or tag, or queries. */
export function gridOption(
  rows: string[],
  values: (number | null)[][], // [row][run]
  labels: string[],
  metric: string,
  { labelCells = true }: { labelCells?: boolean } = {},
): Option {
  const cells: [number, number, number][] = [];
  values.forEach((row, rowIndex) =>
    row.forEach((value, column) => {
      if (typeof value === "number") cells.push([column, rowIndex, value]);
    }),
  );
  return {
    animation: false,
    aria: { enabled: true },
    grid: { left: 16, right: 16, top: 16, bottom: 64, containLabel: true },
    tooltip: { position: "top" },
    xAxis: axis("", {
      type: "category",
      data: labels,
      axisLabel: { color: TEXT, interval: 0, width: 120, overflow: "truncate" },
    }),
    yAxis: axis("", {
      type: "category",
      data: rows,
      inverse: true,
      axisLabel: { color: TEXT, width: 180, overflow: "truncate" },
    }),
    visualMap: {
      min: 0,
      max: 1,
      calculable: false,
      orient: "horizontal",
      left: "center",
      bottom: 0,
      inRange: { color: RAMP },
      textStyle: { color: TEXT },
      text: [`${metric} 1`, "0"],
    },
    series: [
      {
        name: metric,
        type: "heatmap",
        data: cells,
        itemStyle: { borderColor: "transparent", borderWidth: 2 },
        label: {
          show: labelCells,
          formatter: (params: { value: [number, number, number] }) =>
            params.value[2].toFixed(2),
          color: TEXT,
        },
      },
    ],
  };
}

/** Parallel coordinates' dimensions for a comparison: the differing
 * parameters (at most six; the models' recorded fields only when nothing
 * else differs), the metrics, and p95 latency. */
export function parallelDimensions(
  params: { key: string; values: unknown[] }[],
  table: MetricRow[],
  latencyP95: (number | null)[],
): ParallelDimension[] {
  const own = params.filter((param) => !param.key.startsWith("models."));
  const shown = (own.length ? own : params).slice(0, 6);
  return [
    ...shown.map((param) => ({
      name: param.key,
      kind:
        param.values.every((value) => typeof value === "number") &&
        new Set(param.values).size > 2
          ? ("value" as const)
          : ("category" as const),
      values: param.values,
    })),
    ...table.map((row) => ({
      name: row.metric,
      kind: "value" as const,
      values: row.cells.map((cell) => cell.mean ?? null),
    })),
    { name: "p95 ms", kind: "value" as const, values: latencyP95 },
  ];
}
