import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  PALETTE,
  RAMP,
  effectRange,
  effectsOption,
  errorBar,
  heatmapOption,
  labelColor,
  paretoOption,
  whereLostOption,
} from "../src/lab/charts";
import { routingDocument, routingYaml } from "../src/lab/experiments";

type Series = { type: string; data: unknown[] }[];

describe("chart options", () => {
  it("draws effects as bars with their intervals", () => {
    const option = effectsOption(
      [
        { label: "600", delta: 0.08, low: 0.06, high: 0.1, pairs: 2 },
        { label: "800", delta: null, pairs: 0 },
      ],
      "recall@5",
    );
    const series = option.series as Series;
    assert.deepEqual(series[0]?.data, [0.08]);
    assert.deepEqual(series[1]?.data, [[0, 0.06, 0.1]]);
    assert.equal(series[1]?.type, "custom");
    assert.deepEqual((option.xAxis as { data: string[] }).data, ["600"]);
    const drawn = errorBar(null, {
      value: (dimension: number) => [0, 0.06, 0.1][dimension] ?? 0,
      coord: ([x, y]: [number, number]) => [x * 100, 100 - y * 100],
      size: () => [40, 0],
    }) as { children: unknown[] };
    assert.equal(drawn.children.length, 3);
    // The axis holds the whole interval and zero.
    assert.deepEqual(
      effectRange([
        { label: "a", delta: 0.02, low: -0.01, high: 0.05, pairs: 1 },
      ]),
      { min: -0.016, max: 0.056 },
    );
  });

  it("draws interactions on the colour-blind-safe ramp", () => {
    const option = heatmapOption(
      {
        rows: "chunk",
        columns: "rerank",
        row_labels: ["a", "b"],
        column_labels: ["x", "y"],
        cells: [
          [
            { mean: 0.5, trials: 1 },
            { mean: 0.7, trials: 1 },
          ],
          [
            { mean: null, trials: 0 },
            { mean: 0.6, trials: 1 },
          ],
        ],
      },
      "recall@5",
    );
    const series = option.series as Series;
    assert.deepEqual(
      (series[0]?.data as { value: number[] }[]).map((item) => item.value),
      [
        [0, 0, 0.5],
        [1, 0, 0.7],
        [1, 1, 0.6],
      ],
    );
    assert.equal(labelColor(0.5, 0.5, 0.7), "#ffffff");
    assert.equal(labelColor(0.7, 0.5, 0.7), "#1a1a1a");
    const visual = option.visualMap as {
      min: number;
      max: number;
      inRange: { color: string[] };
    };
    assert.deepEqual([visual.min, visual.max], [0.5, 0.7]);
    assert.deepEqual(visual.inRange.color, RAMP);
  });

  it("draws the Pareto frontier as a line", () => {
    const option = paretoOption(
      [
        { ordinal: 0, objective: 0.7, cost: 100, frontier: true },
        { ordinal: 2, objective: 0.6, cost: 300, frontier: false },
        { ordinal: 1, objective: 0.8, cost: 200, frontier: true },
      ],
      "p95 latency (ms)",
      "recall@5",
    );
    const series = option.series as Series;
    assert.deepEqual(series[0]?.data, [[300, 0.6, 2]]);
    assert.deepEqual(series[1]?.data, [
      [100, 0.7, 0],
      [200, 0.8, 1],
    ]);
    assert.deepEqual(option.color, PALETTE);
  });

  it("stacks where queries were lost", () => {
    const option = whereLostOption([
      { ordinal: 3, found: 50, parse: 1, retrieval: 6, rerank: 2 },
    ]);
    const series = option.series as Series;
    assert.deepEqual(
      series.map((item) => item.data),
      [[50], [1], [6], [2]],
    );
  });
});

describe("routing", () => {
  const route = {
    path: 'parse.route."application/pdf"',
    best: {
      "application/pdf": {
        level: 1,
        label: "parse/docling",
        value: "parse/docling",
        mean: 0.8,
        trials: 2,
      },
      unknown: { level: 0, label: "x", value: "x", mean: 0.1, trials: 1 },
      "text/html": {
        level: 0,
        label: "parse/a",
        value: { component: "parse/a", params: { ocr: true } },
        mean: 0.5,
        trials: 2,
      },
    },
  };

  it("shows the routes as the parse stage's YAML", () => {
    assert.equal(
      routingYaml(route),
      [
        '# The best parse.route."application/pdf" per gold file type',
        "parse:",
        "  route:",
        "    application/pdf: parse/docling",
        '    text/html: {"component":"parse/a","params":{"ocr":true}}',
      ].join("\n"),
    );
  });

  it("applies them to a copy of the base pipeline", () => {
    const base = {
      schema: "stratafi.pipeline.v1",
      name: "base",
      stages: { parse: { route: { default: { component: "parse/x" } } } },
    };
    const document = routingDocument(base, route, "pdf-parsers");
    assert.equal(document.name, "pdf-parsers-routing");
    const routes = (
      document.stages as { parse: { route: Record<string, unknown> } }
    ).parse.route;
    assert.deepEqual(Object.keys(routes).sort(), [
      "application/pdf",
      "default",
      "text/html",
    ]);
    assert.deepEqual(base.stages.parse.route, {
      default: { component: "parse/x" },
    });
  });
});
