// ECharts options for the experiment views: bars
// with error bars, heatmaps and scatter plots with a frontier. Pure functions
// of the API's numbers, so they're tested without a browser. Colours come from
// one colour-blind-safe palette (Okabe and Ito), and text uses currentColor,
// so charts follow the light or dark theme.

export const PALETTE = [
  "#0072B2",
  "#E69F00",
  "#009E73",
  "#CC79A7",
  "#56B4E9",
  "#D55E00",
  "#F0E442",
];
// A sequential, colour-blind-safe ramp (viridis) for heatmaps.
export const RAMP = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"];
const MUTED = "#8a8a8a";
const TEXT = "currentColor";

type Option = Record<string, unknown>;

/** A value or category axis with its name under it. */
function axis(name: string, extra: Option = {}): Option {
  return {
    name,
    nameLocation: "middle",
    nameGap: 32,
    nameTextStyle: { color: TEXT },
    axisLabel: { color: TEXT },
    axisLine: { lineStyle: { color: MUTED } },
    splitLine: { lineStyle: { color: MUTED, opacity: 0.25 } },
    ...extra,
  };
}

export interface EffectRow {
  label: string;
  delta?: number | null;
  low?: number | null;
  high?: number | null;
  pairs: number;
}

/** A vertical axis, its name above it reading rightwards, so long names fit. */
function verticalAxis(name: string, extra: Option = {}): Option {
  return axis(name, {
    nameLocation: "end",
    nameGap: 12,
    nameTextStyle: { color: TEXT, align: "left" },
    ...extra,
  });
}

/** One factor's effects: a bar per level with its 95% interval as a whisker. */
export function effectsOption(rows: EffectRow[], metric: string): Option {
  const shown = rows.filter((row) => typeof row.delta === "number");
  return {
    animation: false,
    aria: { enabled: true },
    color: PALETTE,
    grid: { left: 64, right: 16, top: 16, bottom: 56, containLabel: true },
    tooltip: { trigger: "axis" },
    xAxis: axis("Level", {
      type: "category",
      data: shown.map((row) => row.label),
      axisLabel: {
        color: TEXT,
        interval: 0,
        rotate: shown.length > 3 ? 20 : 0,
      },
    }),
    yAxis: verticalAxis(`Change in ${metric}`, {
      type: "value",
      ...effectRange(shown),
    }),
    series: [
      {
        name: "Effect",
        type: "bar",
        data: shown.map((row) => row.delta),
        barMaxWidth: 48,
      },
      {
        name: "95% CI",
        type: "custom",
        data: shown.map((row, index) => [index, row.low, row.high]),
        renderItem: errorBar,
        z: 3,
      },
    ],
  };
}

/** An axis range holding every interval and zero, padded a little. */
export function effectRange(rows: EffectRow[]): { min: number; max: number } {
  const ends = rows.flatMap((row) =>
    [row.low, row.high, row.delta].filter(
      (value): value is number => typeof value === "number",
    ),
  );
  const low = Math.min(0, ...ends);
  const high = Math.max(0, ...ends);
  const pad = (high - low || 1) * 0.1;
  const round = (value: number) => Number(value.toPrecision(3));
  return { min: round(low - pad), max: round(high + pad) };
}

interface RenderApi {
  value(dimension: number): number;
  coord(point: [number, number]): [number, number];
  size(size: [number, number]): [number, number];
}

/** ECharts' custom series: a whisker from low to high with caps. */
export function errorBar(_params: unknown, api: RenderApi): Option {
  const index = api.value(0);
  const low = api.coord([index, api.value(1)]);
  const high = api.coord([index, api.value(2)]);
  const cap = Math.min(8, api.size([1, 0])[0] * 0.15);
  const style = { stroke: TEXT, lineWidth: 1.5 };
  return {
    type: "group",
    children: [
      {
        type: "line",
        shape: { x1: low[0], y1: low[1], x2: high[0], y2: high[1] },
        style,
      },
      {
        type: "line",
        shape: { x1: low[0] - cap, y1: low[1], x2: low[0] + cap, y2: low[1] },
        style,
      },
      {
        type: "line",
        shape: {
          x1: high[0] - cap,
          y1: high[1],
          x2: high[0] + cap,
          y2: high[1],
        },
        style,
      },
    ],
  };
}

export interface Interaction {
  rows: string;
  columns: string;
  row_labels: string[];
  column_labels: string[];
  cells: { mean: number | null; trials: number }[][];
}

/** A pair of factors: each cell's mean objective, on the sequential ramp. */
export function heatmapOption(
  interaction: Interaction,
  metric: string,
): Option {
  const cells: [number, number, number][] = [];
  interaction.cells.forEach((row, rowIndex) =>
    row.forEach((cell, columnIndex) => {
      if (cell.mean !== null) cells.push([columnIndex, rowIndex, cell.mean]);
    }),
  );
  const values = cells.map((cell) => cell[2]);
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 1;
  // Light text on the ramp's dark end, dark text on its light end.
  const data = cells.map((value) => ({
    value,
    label: { color: labelColor(value[2], min, max) },
  }));
  return {
    animation: false,
    aria: { enabled: true },
    grid: { left: 16, right: 16, top: 16, bottom: 72, containLabel: true },
    tooltip: { position: "top" },
    xAxis: axis(interaction.columns, {
      type: "category",
      data: interaction.column_labels,
      axisLabel: { color: TEXT, interval: 0 },
    }),
    yAxis: verticalAxis(interaction.rows, {
      type: "category",
      data: interaction.row_labels,
    }),
    visualMap: {
      min,
      max: max > min ? max : min + 1e-6,
      calculable: false,
      orient: "horizontal",
      left: "center",
      bottom: 0,
      inRange: { color: RAMP },
      textStyle: { color: TEXT },
      text: [`${metric} high`, "low"],
    },
    series: [
      {
        name: metric,
        type: "heatmap",
        data,
        label: {
          show: true,
          formatter: (params: { value: [number, number, number] }) =>
            params.value[2].toFixed(3),
        },
      },
    ],
  };
}

/** A heatmap cell's label colour: readable on its place in the ramp. */
export function labelColor(value: number, min: number, max: number): string {
  const position = max > min ? (value - min) / (max - min) : 0;
  return position > 0.55 ? "#1a1a1a" : "#ffffff";
}

export interface ParetoPoint {
  ordinal: number;
  objective: number;
  cost: number;
  frontier: boolean;
}

/** The objective against a cost; the frontier's points joined by a line. */
export function paretoOption(
  points: ParetoPoint[],
  costLabel: string,
  metric: string,
): Option {
  const frontier = points
    .filter((point) => point.frontier)
    .sort((a, b) => a.cost - b.cost);
  return {
    animation: false,
    aria: { enabled: true },
    color: PALETTE,
    grid: { left: 64, right: 24, top: 24, bottom: 56, containLabel: true },
    tooltip: { trigger: "item" },
    legend: { textStyle: { color: TEXT }, top: 0 },
    xAxis: axis(costLabel, { type: "value", scale: true }),
    yAxis: verticalAxis(metric, { type: "value", scale: true }),
    series: [
      {
        name: "Trials",
        type: "scatter",
        symbolSize: 10,
        data: points
          .filter((point) => !point.frontier)
          .map((point) => [point.cost, point.objective, point.ordinal]),
      },
      {
        name: "Frontier",
        type: "line",
        symbolSize: 12,
        data: frontier.map((point) => [
          point.cost,
          point.objective,
          point.ordinal,
        ]),
      },
    ],
  };
}

export interface LostRow {
  ordinal: number;
  found: number;
  parse: number;
  retrieval: number;
  rerank: number;
}

const LOST_STAGES: [keyof LostRow, string][] = [
  ["found", "Found"],
  ["parse", "Lost at parse"],
  ["retrieval", "Lost at retrieval"],
  ["rerank", "Lost at rerank"],
];

/** Per trial, its queries found and lost by stage, stacked. */
export function whereLostOption(rows: LostRow[]): Option {
  return {
    animation: false,
    aria: { enabled: true },
    color: [PALETTE[2], PALETTE[5], PALETTE[1], PALETTE[3]],
    grid: { left: 48, right: 16, top: 32, bottom: 40, containLabel: true },
    tooltip: { trigger: "axis" },
    legend: { textStyle: { color: TEXT }, top: 0 },
    xAxis: axis("Trial", {
      type: "category",
      data: rows.map((row) => `#${row.ordinal}`),
    }),
    yAxis: verticalAxis("Queries", { type: "value" }),
    series: LOST_STAGES.map(([key, name]) => ({
      name,
      type: "bar",
      stack: "queries",
      data: rows.map((row) => row[key]),
    })),
  };
}
