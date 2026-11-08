// Traces and latency profiles: a query's retrieval as bars on a
// timeline, and each step's distribution as box plots, one per run. Pure
// functions of the API's numbers; IDs and numbers only.

import { PALETTE } from "./charts";
import { STEP_NAMES } from "./compare";
import { runColor } from "./tracking";

type Option = Record<string, unknown>;
const TEXT = "currentColor";
const MUTED = "#8a8a8a";

export interface Span {
  id: string;
  name: string;
  kind: string;
  start_ms: number;
  duration_ms: number;
}

const KIND_COLORS: Record<string, string> = {
  search: MUTED,
  retriever: PALETTE[0] as string,
  embedding: PALETTE[1] as string,
  fusion: PALETTE[2] as string,
  reranker: PALETTE[3] as string,
  other: MUTED,
};

export function spanColor(kind: string): string {
  return KIND_COLORS[kind] ?? MUTED;
}

interface RenderApi {
  value(dimension: number): number;
  coord(point: [number, number]): [number, number];
  size(size: [number, number]): [number, number];
  visual(name: string): string;
}

/** A span's bar from its start to its end, in its row. */
export function spanBar(_params: unknown, api: RenderApi): Option {
  const row = api.value(0);
  const [x1, y] = api.coord([api.value(1), row]);
  const [x2] = api.coord([api.value(2), row]);
  const height = Math.abs(api.size([0, 1])[1]) * 0.55;
  return {
    type: "rect",
    shape: {
      x: x1,
      y: y - height / 2,
      width: Math.max(x2 - x1, 2),
      height,
      r: 3,
    },
    style: { fill: api.visual("color") },
  };
}

/** The spans on a timeline, in the order they ran: a waterfall. */
export function waterfallOption(spans: Span[]): Option {
  return {
    animation: false,
    aria: { enabled: true },
    grid: { left: 16, right: 40, top: 16, bottom: 40, containLabel: true },
    tooltip: {
      trigger: "item",
      formatter: (params: { dataIndex: number }) => {
        const span = spans[params.dataIndex];
        if (!span) return "";
        return `${span.name}: ${span.duration_ms.toFixed(2)} ms, from ${span.start_ms.toFixed(2)} ms`;
      },
    },
    xAxis: {
      type: "value",
      min: 0,
      name: "ms from the search's start",
      nameLocation: "middle",
      nameGap: 28,
      nameTextStyle: { color: TEXT },
      axisLabel: { color: TEXT },
      splitLine: { lineStyle: { color: MUTED, opacity: 0.25 } },
    },
    yAxis: {
      type: "category",
      data: spans.map((span) => span.name),
      inverse: true,
      axisLabel: { color: TEXT },
      axisLine: { lineStyle: { color: MUTED } },
    },
    series: [
      {
        name: "Spans",
        type: "custom",
        renderItem: spanBar,
        encode: { x: [1, 2], y: 0 },
        data: spans.map((span, index) => ({
          value: [index, span.start_ms, span.start_ms + span.duration_ms],
          itemStyle: { color: spanColor(span.kind) },
        })),
      },
    ],
  };
}

export interface StepProfile {
  min: number;
  p25: number;
  p50: number;
  p75: number;
  p95: number;
}

export const PROFILE_STEPS = [
  "keyword_ms",
  "embed_ms",
  "dense_ms",
  "fusion_ms",
  "rerank_ms",
  "total_ms",
];

/** Box plots of each step's latency, one series per run: the box from p25 to
 * p75 with the median, whiskers from the minimum to p95. */
export function stepBoxOption(
  profiles: (Record<string, StepProfile> | null)[],
  labels: string[],
): Option {
  const steps = PROFILE_STEPS.filter((step) =>
    profiles.some((profile) => profile && step in profile),
  );
  return {
    animation: false,
    aria: { enabled: true },
    legend: { data: labels, textStyle: { color: TEXT }, top: 0 },
    grid: { left: 16, right: 24, top: 36, bottom: 40, containLabel: true },
    tooltip: { trigger: "item" },
    xAxis: {
      type: "category",
      data: steps.map((step) => STEP_NAMES[step] ?? "Total"),
      axisLabel: { color: TEXT, interval: 0, hideOverlap: false, fontSize: 11 },
      axisLine: { lineStyle: { color: MUTED } },
    },
    yAxis: {
      type: "value",
      name: "ms",
      nameTextStyle: { color: TEXT },
      axisLabel: { color: TEXT },
      splitLine: { lineStyle: { color: MUTED, opacity: 0.25 } },
    },
    series: labels.map((label, index) => ({
      name: label,
      type: "boxplot",
      color: runColor(index),
      itemStyle: {
        color: "transparent",
        borderColor: runColor(index),
        borderWidth: 2,
      },
      data: steps.map((step) => {
        const found = profiles[index]?.[step];
        return found
          ? [found.min, found.p25, found.p50, found.p75, found.p95]
          : [];
      }),
    })),
  };
}

/** One step's histogram over a run's queries. */
export function histogramOption(
  histogram: { edges: number[]; counts: number[] },
  step: string,
): Option {
  const labels = histogram.counts.map((_, index) => {
    const low = histogram.edges[index] ?? 0;
    const high = histogram.edges[index + 1] ?? low;
    return `${low.toFixed(1)}–${high.toFixed(1)}`;
  });
  return {
    animation: false,
    aria: { enabled: true },
    color: [PALETTE[0]],
    grid: { left: 16, right: 16, top: 32, bottom: 64, containLabel: true },
    tooltip: { trigger: "axis" },
    xAxis: {
      type: "category",
      data: labels,
      name: `${STEP_NAMES[step] ?? "Total"} (ms)`,
      nameLocation: "middle",
      nameGap: 52,
      nameTextStyle: { color: TEXT },
      axisLabel: { color: TEXT, rotate: 30 },
    },
    yAxis: {
      type: "value",
      name: "Queries",
      nameTextStyle: { color: TEXT },
      axisLabel: { color: TEXT },
      splitLine: { lineStyle: { color: MUTED, opacity: 0.25 } },
    },
    series: [
      {
        name: "Queries",
        type: "bar",
        barCategoryGap: "8%",
        data: histogram.counts,
      },
    ],
  };
}
