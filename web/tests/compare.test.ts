import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { RAMP } from "../src/lab/charts";
import {
  curvesOption,
  forestMark,
  forestOption,
  funnelOption,
  gridOption,
  latencyOption,
  lostOption,
  parallelDimensions,
  parallelOption,
  qualityLatencyOption,
  runLabels,
} from "../src/lab/compare";
import {
  diffMaps,
  edgePath,
  layoutMap,
  mapRows,
  nodeSubtitle,
  overlayLines,
  overlaySummary,
  visibleEdges,
  type MapLike,
} from "../src/lab/maps";
import {
  histogramOption,
  spanBar,
  spanColor,
  stepBoxOption,
  waterfallOption,
} from "../src/lab/traces";
import { runColor } from "../src/lab/tracking";

type Series = {
  type: string;
  data: unknown[];
  name?: string;
  color?: string;
  symbol?: string;
}[];

const LABELS = runLabels([{ label: "hybrid" }, { label: "no rerank" }]);

describe("compare charts", () => {
  it("name runs by their place, so two with one title differ", () => {
    assert.deepEqual(LABELS, ["1. hybrid", "2. no rerank"]);
    assert.deepEqual(
      runLabels([{ label: "a very long title for a run" }], 12),
      ["1. a very l…"],
    );
  });

  it("leave the models' mirror fields out of parallel coordinates", () => {
    const dimensions = parallelDimensions(
      [
        {
          key: "models.rerank.component",
          values: ["rerank/gateway", "rerank/none"],
        },
        { key: "rerank.component", values: ["rerank/gateway", "rerank/none"] },
      ],
      [],
      [1, 2],
    );
    assert.deepEqual(
      dimensions.map((dimension) => dimension.name),
      ["rerank.component", "p95 ms"],
    );
    const only = parallelDimensions(
      [{ key: "models.embed.revision", values: ["r1", "r2"] }],
      [],
      [1, 2],
    );
    assert.equal(only[0]?.name, "models.embed.revision");
  });

  it("draw every metric's interval per run, each run in its own colour", () => {
    const option = forestOption(
      [
        {
          metric: "recall@5",
          cells: [
            { mean: 0.8, low: 0.7, high: 0.9 },
            { mean: 0.6, low: 0.5, high: 0.7 },
          ],
        },
        { metric: "mrr@10", cells: [{ mean: 0.5, low: 0.4, high: 0.6 }, {}] },
      ],
      LABELS,
    );
    const series = option.series as Series;
    assert.equal(series.length, 2);
    assert.equal(series[0]?.color, runColor(0));
    assert.deepEqual(series[0]?.data, [
      [0, 0.8, 0.7, 0.9],
      [1, 0.5, 0.4, 0.6],
    ]);
    // The second run has no mrr@10: no mark rather than a zero.
    assert.deepEqual(series[1]?.data, [[0, 0.6, 0.5, 0.7]]);
    const axis = option.xAxis as { min: number; max: number };
    assert.ok(axis.min <= 0.4 && axis.max >= 0.9 && axis.max <= 1);
    const drawn = forestMark(2)(
      { seriesIndex: 1 },
      {
        value: (dimension: number) => [0, 0.6, 0.5, 0.7][dimension] ?? 0,
        coord: ([x, y]: [number, number]) => [x * 100, y * 30 + 15],
        size: () => [0, 30],
        visual: () => "#000",
      },
    ) as { children: { shape: { y1?: number; cy?: number } }[] };
    // The second of two runs sits below the band's middle (15), by a third of
    // the band's height (30) over two.
    assert.equal(drawn.children[1]?.shape.cy, 20);
  });

  it("draw curves, the funnel, where lost and latency", () => {
    const curves = curvesOption(
      {
        metric: "recall",
        ks: ["1", "5", "all"],
        series: [
          [0.4, 0.8, 0.9],
          [0.3, 0.7, null],
        ],
      },
      LABELS,
    );
    assert.deepEqual((curves.xAxis as { data: string[] }).data, [
      "@1",
      "@5",
      "all",
    ]);
    assert.notEqual(
      (curves.series as Series)[0]?.symbol,
      (curves.series as Series)[1]?.symbol,
    );
    const funnel = funnelOption(
      {
        checkpoints: ["dense", "fused", "final"],
        hit: [
          [0.9, 0.95, 0.8],
          [0.9, 0.95, 0.7],
        ],
      },
      LABELS,
    );
    assert.deepEqual((funnel.xAxis as { data: string[] }).data, [
      "Dense",
      "Fused",
      "Final",
    ]);
    const lost = lostOption(
      {
        buckets: ["found", "parse", "retrieval", "rerank", "filtered"],
        counts: [
          [50, 1, 6, 3, 0],
          [45, 1, 6, 8, 0],
        ],
      },
      LABELS,
    );
    const lostSeries = lost.series as Series;
    assert.deepEqual(
      lostSeries.map((series) => series.name),
      ["Found", "Lost at parse", "Lost at retrieval", "Lost at rerank"],
    );
    assert.deepEqual(lostSeries[3]?.data, [3, 8]);
    const latency = latencyOption(
      {
        steps: ["embed_ms", "rerank_ms", "total_ms"],
        p50: [
          [1, 20, 22],
          [1, 0, 2],
        ],
      },
      LABELS,
    );
    const latencySeries = latency.series as Series;
    assert.deepEqual(
      latencySeries.map((series) => series.name),
      ["Embed query", "Rerank"],
    );
    assert.deepEqual(latencySeries[1]?.data, [20, 0]);
  });

  it("mark the runs no other run beats on quality and latency", () => {
    const option = qualityLatencyOption(
      [
        { label: "a", quality: 0.8, latency: 30 },
        { label: "b", quality: 0.7, latency: 5 },
        { label: "c", quality: 0.6, latency: 40 },
        { label: "d", quality: null, latency: 10 },
      ],
      "recall@5",
    );
    const series = option.series as Series;
    const frontier = series.find((item) => item.name === "Frontier");
    assert.deepEqual(frontier?.data, [
      [5, 0.7],
      [30, 0.8],
    ]);
    assert.equal(series.filter((item) => item.type === "scatter").length, 3);
  });

  it("build parallel coordinates from the differing parameters and the metrics", () => {
    const dimensions = parallelDimensions(
      [
        { key: "rerank.component", values: ["rerank/gateway", "rerank/none"] },
        { key: "top_k", values: [8, 5] },
      ],
      [{ metric: "recall@5", cells: [{ mean: 0.8 }, { mean: 0.7 }] }],
      [30, 5],
    );
    assert.deepEqual(
      dimensions.map((dimension) => [dimension.name, dimension.kind]),
      [
        ["rerank.component", "category"],
        ["top_k", "category"],
        ["recall@5", "value"],
        ["p95 ms", "value"],
      ],
    );
    const option = parallelOption(dimensions, LABELS);
    const series = option.series as Series;
    assert.deepEqual(series[0]?.data, [["rerank/gateway", "8", 0.8, 30]]);
    const axes = option.parallelAxis as { type: string; data?: string[] }[];
    assert.deepEqual(axes[0]?.data, ["rerank/gateway", "rerank/none"]);
  });

  it("draw slices and queries on the sequential ramp", () => {
    const option = gridOption(
      ["pdf", "docx"],
      [
        [0.5, 0.6],
        [null, 0.9],
      ],
      LABELS,
      "recall@5",
    );
    assert.deepEqual((option.series as Series)[0]?.data, [
      [0, 0, 0.5],
      [1, 0, 0.6],
      [1, 1, 0.9],
    ]);
    assert.deepEqual(
      (option.visualMap as { inRange: { color: string[] } }).inRange.color,
      RAMP,
    );
  });
});

describe("traces", () => {
  const spans = [
    {
      id: "search",
      name: "Search",
      kind: "search",
      start_ms: 0,
      duration_ms: 10,
    },
    {
      id: "keyword",
      name: "Keyword search",
      kind: "retriever",
      start_ms: 0,
      duration_ms: 2,
    },
    {
      id: "rerank",
      name: "Rerank",
      kind: "reranker",
      start_ms: 2,
      duration_ms: 8,
    },
  ];

  it("draw spans as bars from their start to their end", () => {
    const option = waterfallOption(spans);
    const data = (option.series as { data: { value: number[] }[] }[])[0]?.data;
    assert.deepEqual(
      data?.map((item) => item.value),
      [
        [0, 0, 10],
        [1, 0, 2],
        [2, 2, 10],
      ],
    );
    // ECharts picks a round end for the axis.
    assert.equal((option.xAxis as { max?: number }).max, undefined);
    assert.notEqual(spanColor("retriever"), spanColor("reranker"));
    const bar = spanBar(null, {
      value: (dimension: number) => [2, 2, 10][dimension] ?? 0,
      coord: ([x, y]: [number, number]) => [x * 10, y * 20 + 10],
      size: () => [0, 20],
      visual: () => "#000",
    }) as { shape: { x: number; width: number } };
    assert.deepEqual([bar.shape.x, bar.shape.width], [20, 80]);
  });

  it("draw each run's steps as box plots, and a step's histogram", () => {
    const step = { min: 1, p25: 2, p50: 3, p75: 4, p95: 9 };
    const option = stepBoxOption(
      [{ rerank_ms: step, total_ms: step }, null],
      LABELS,
    );
    const series = option.series as Series;
    assert.deepEqual((option.xAxis as { data: string[] }).data, [
      "Rerank",
      "Total",
    ]);
    assert.deepEqual(series[0]?.data, [
      [1, 2, 3, 4, 9],
      [1, 2, 3, 4, 9],
    ]);
    assert.deepEqual(series[1]?.data, [[], []]);
    const histogram = histogramOption(
      { edges: [0, 5, 10], counts: [3, 1] },
      "rerank_ms",
    );
    assert.deepEqual((histogram.xAxis as { data: string[] }).data, [
      "0.0–5.0",
      "5.0–10.0",
    ]);
  });
});

const MAP: MapLike = {
  nodes: [
    { id: "extract", kind: "stage", lane: "ingest", label: "Upload", column: 0, row: 0, component: "extract/upload", params_hash: "a" },
    { id: "parse:default", kind: "stage", lane: "ingest", label: "Parser", column: 1, row: 0, component: "parse/markdown", params_hash: "b" },
    { id: "parse:pdf", kind: "stage", lane: "ingest", label: "Docling", column: 1, row: 1, component: "parse/docling", params_hash: "c" },
    { id: "query:rerank", kind: "query", lane: "query", label: "Rerank", column: 4, row: 0, component: "rerank/gateway", params_hash: "d" },
    { id: "store:objects", kind: "datastore", lane: "stores", label: "Artifact store", column: 0, row: 0, backend: "s3", configured: true },
    { id: "service:model-gateway", kind: "service", lane: "services", label: "Model gateway", column: 0, row: 0, configured: false },
  ],
  edges: [
    { id: "e1", source: "extract", target: "parse:default", kind: "flow", label: "raw", sends_text: false },
    { id: "e2", source: "extract", target: "store:objects", kind: "write", label: "its artifact", sends_text: false },
    { id: "e3", source: "parse:default", target: "store:objects", kind: "write", label: "its artifact", sends_text: false },
    { id: "e4", source: "store:objects", target: "query:rerank", kind: "read", label: "index", sends_text: false },
    { id: "e5", source: "query:rerank", target: "service:model-gateway", kind: "call", label: "query and passage text", sends_text: true },
  ],
}; // prettier-ignore

describe("pipeline maps", () => {
  it("bundle ingest steps' store edges unless asked or selected", () => {
    const bundled = visibleEdges(MAP);
    assert.deepEqual(
      bundled.edges.map((edge) => edge.id),
      ["e1", "e4", "e5"],
    );
    assert.deepEqual(bundled.bundles, [{ store: "store:objects", count: 2 }]);
    const selected = visibleEdges(MAP, { selected: "extract" });
    assert.deepEqual(
      selected.edges.map((edge) => edge.id),
      ["e1", "e2", "e4", "e5"],
    );
    assert.equal(visibleEdges(MAP, { showAll: true }).edges.length, 5);
  });

  it("lay out lanes top to bottom, routes in one column, stores under their steps", () => {
    const layout = layoutMap(MAP);
    assert.deepEqual(
      layout.lanes.map((lane) => lane.lane),
      ["services", "ingest", "query", "stores"],
    );
    const box = (id: string) => layout.boxes.find((item) => item.id === id);
    assert.equal(box("parse:default")?.x, box("parse:pdf")?.x);
    assert.ok((box("parse:pdf")?.y ?? 0) > (box("parse:default")?.y ?? 0));
    assert.ok((box("store:objects")?.y ?? 0) > (box("query:rerank")?.y ?? 0));
    assert.ok(
      (box("service:model-gateway")?.y ?? 1) < (box("extract")?.y ?? 0),
    );
    const bundle = layout.edges.find((edge) => edge.bundle);
    assert.equal(bundle?.label, "2 reads and writes by ingest steps");
    const call = layout.edges.find((edge) => edge.id === "e5");
    assert.equal(call?.sendsText, true);
    assert.ok(layout.width > 0 && layout.height > 0);
    const highlighted = layoutMap(MAP, { selected: "extract" }).edges.filter(
      (edge) => edge.highlighted,
    );
    assert.deepEqual(highlighted.map((edge) => edge.id).sort(), ["e1", "e2"]);
  });

  it("route edges side to side in a lane and top to bottom between lanes", () => {
    const layout = layoutMap(MAP);
    const box = (id: string) => layout.boxes.find((item) => item.id === id);
    const across = edgePath(box("extract")!, box("parse:default")!);
    assert.match(across.path, /^M 188 /);
    const down = edgePath(box("query:rerank")!, box("store:objects")!);
    assert.ok(down.y > (box("query:rerank")?.y ?? 0));
  });

  it("keep a read and a write between one pair of boxes apart", () => {
    const paired: MapLike = {
      nodes: MAP.nodes,
      edges: [
        ...MAP.edges,
        { id: "e6", source: "store:objects", target: "extract", kind: "read", label: "input artifacts", sends_text: false },
      ],
    }; // prettier-ignore
    const layout = layoutMap(paired, { showAll: true });
    const write = layout.edges.find((edge) => edge.id === "e2");
    const read = layout.edges.find((edge) => edge.id === "e6");
    assert.ok(write && read);
    assert.notEqual(write.path, read.path);
    assert.notEqual(write.labelY, read.labelY);
  });

  it("diff two maps node by node", () => {
    const other: MapLike = {
      nodes: [
        ...MAP.nodes.filter((node) => node.id !== "parse:pdf"),
        { id: "parse:default", kind: "stage", lane: "ingest", label: "Parser", column: 1, row: 0, component: "parse/markdown", params_hash: "changed" },
        { id: "clean:0", kind: "stage", lane: "ingest", label: "Whitespace", column: 2, row: 0, component: "clean/whitespace" },
      ],
      edges: [],
    }; // prettier-ignore
    const changes = diffMaps(MAP, other);
    assert.equal(changes["parse:default"], "changed");
    assert.equal(changes["clean:0"], "added");
    assert.equal(changes["parse:pdf"], "removed");
    assert.equal(changes.extract, "same");
  });

  it("summarize a node's numbers and name what it is", () => {
    assert.equal(
      overlaySummary({
        tasks: 12,
        cache_hits: 3,
        duration_ms: 1500,
        bytes: 2048,
      }),
      "12 tasks · 3 cached · 1500 ms · 2.0 KB",
    );
    assert.equal(
      overlaySummary({
        latency_ms: { p50: 3, p95: 9.5 },
        evidence_reaches: { fused: 0.9, candidates: 0.85 },
      }),
      "p50 3.0 ms, p95 9.5 ms · evidence 85%",
    );
    assert.equal(overlaySummary(null), "");
    assert.deepEqual(
      overlayLines({
        tasks: 12,
        cache_hits: 3,
        duration_ms: 1500,
        bytes: 2048,
      }),
      ["12 tasks · 3 cached", "1500 ms · 2.0 KB"],
    );
    assert.deepEqual(overlayLines(null), []);
    assert.equal(nodeSubtitle(MAP.nodes[4]!), "s3");
    assert.equal(nodeSubtitle(MAP.nodes[5]!), "not configured here");
    assert.equal(nodeSubtitle(MAP.nodes[0]!), "extract/upload");
    const rows = mapRows(MAP);
    assert.equal(rows.nodes.length, 6);
    assert.deepEqual(rows.edges[4], [
      "Rerank",
      "call",
      "Model gateway",
      "query and passage text (text leaves the lab)",
    ]);
  });
});
