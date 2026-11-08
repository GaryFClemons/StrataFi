// Model comparison charts: each model's best run with its interval,
// and the matrix of controlled pairs on a diverging scale. Pure functions.

import { PALETTE } from "./charts";

type Option = Record<string, unknown>;
const TEXT = "currentColor";
const MUTED = "#8a8a8a";
// Two opposite hues and a neutral grey midpoint: worse, no difference, better.
export const DIVERGING = [
  "#D55E00",
  "#E8B191",
  "#B8B8B8",
  "#8DB8D8",
  "#0072B2",
];

export interface ModelRow {
  label: string;
  best?: { mean?: number; low?: number; high?: number } | null;
}

interface RenderApi {
  value(dimension: number): number;
  coord(point: [number, number]): [number, number];
  size(size: [number, number]): [number, number];
}

/** A horizontal whisker from low to high, with caps. */
export function horizontalWhisker(_params: unknown, api: RenderApi): Option {
  const row = api.value(0);
  const [x1, y] = api.coord([api.value(1), row]);
  const [x2] = api.coord([api.value(2), row]);
  const cap = Math.min(7, Math.abs(api.size([0, 1])[1]) * 0.18);
  const style = { stroke: TEXT, lineWidth: 1.5 };
  return {
    type: "group",
    children: [
      { type: "line", shape: { x1, y1: y, x2, y2: y }, style },
      { type: "line", shape: { x1, y1: y - cap, x2: x1, y2: y + cap }, style },
      { type: "line", shape: { x1: x2, y1: y - cap, x2, y2: y + cap }, style },
    ],
  };
}

/** Each model's best run, best first, as a bar with its 95% interval. */
export function modelBarsOption(rows: ModelRow[], metric: string): Option {
  const shown = rows.filter((row) => typeof row.best?.mean === "number");
  return {
    animation: false,
    aria: { enabled: true },
    color: [PALETTE[0]],
    grid: { left: 16, right: 24, top: 16, bottom: 40, containLabel: true },
    tooltip: { trigger: "axis" },
    xAxis: {
      type: "value",
      min: 0,
      max: 1,
      name: `Best run's ${metric}`,
      nameLocation: "middle",
      nameGap: 28,
      nameTextStyle: { color: TEXT },
      axisLabel: { color: TEXT },
      splitLine: { lineStyle: { color: MUTED, opacity: 0.25 } },
    },
    yAxis: {
      type: "category",
      data: shown.map((row) => row.label),
      inverse: true,
      axisLabel: { color: TEXT, width: 200, overflow: "truncate" },
    },
    series: [
      {
        name: metric,
        type: "bar",
        barMaxWidth: 22,
        itemStyle: { borderRadius: [0, 4, 4, 0] },
        data: shown.map((row) => row.best?.mean),
      },
      {
        name: "95% CI",
        type: "custom",
        renderItem: horizontalWhisker,
        encode: { x: [1, 2], y: 0 },
        data: shown.map((row, index) => [
          index,
          row.best?.low ?? row.best?.mean,
          row.best?.high ?? row.best?.mean,
        ]),
        z: 3,
      },
    ],
  };
}

/** The controlled pairs' mean differences: cell (row i, column j) is model j
 * minus model i, on a diverging scale centred on no difference. */
export function pairMatrixOption(
  labels: string[],
  matrix: ({ delta: number; pairs: number } | null)[][],
  metric: string,
): Option {
  const cells: { value: [number, number, number]; pairs: number }[] = [];
  matrix.forEach((row, i) =>
    row.forEach((cell, j) => {
      if (cell) cells.push({ value: [j, i, cell.delta], pairs: cell.pairs });
    }),
  );
  const extent = Math.max(
    0.01,
    ...cells.map((cell) => Math.abs(cell.value[2])),
  );
  return {
    animation: false,
    aria: { enabled: true },
    grid: { left: 16, right: 16, top: 16, bottom: 72, containLabel: true },
    tooltip: { position: "top" },
    xAxis: {
      type: "category",
      data: labels,
      name: "Model compared (B)",
      nameLocation: "middle",
      nameGap: 28,
      nameTextStyle: { color: TEXT },
      axisLabel: { color: TEXT, interval: 0, width: 110, overflow: "truncate" },
    },
    yAxis: {
      type: "category",
      data: labels,
      inverse: true,
      name: "Against (A)",
      nameTextStyle: { color: TEXT },
      axisLabel: { color: TEXT, width: 140, overflow: "truncate" },
    },
    visualMap: {
      min: -extent,
      max: extent,
      calculable: false,
      orient: "horizontal",
      left: "center",
      bottom: 0,
      inRange: { color: DIVERGING },
      textStyle: { color: TEXT },
      text: [`B better on ${metric}`, "B worse"],
    },
    series: [
      {
        name: `${metric}: B minus A`,
        type: "heatmap",
        data: cells,
        label: {
          show: true,
          color: "#1a1a1a",
          formatter: (params: { value: [number, number, number] }) =>
            `${params.value[2] >= 0 ? "+" : ""}${params.value[2].toFixed(3)}`,
        },
        itemStyle: { borderColor: "transparent", borderWidth: 2 },
      },
    ],
  };
}
