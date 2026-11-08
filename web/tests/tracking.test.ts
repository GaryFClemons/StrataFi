import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { parseInline, parseMarkdown } from "../src/lab/markdown";
import {
  DIVERGING,
  horizontalWhisker,
  modelBarsOption,
  pairMatrixOption,
} from "../src/lab/models";
import {
  MAX_COMPARED,
  RUN_COLORS,
  columnChoices,
  columnLabel,
  formatTags,
  formatValue,
  groupRows,
  headlineRecall,
  parseTags,
  runColor,
  runSymbol,
  toggleSelection,
} from "../src/lab/tracking";

type Series = { type: string; data: unknown[]; name?: string }[];

describe("the tracking table", () => {
  it("gives each run a colour and a shape by its place, never its rank", () => {
    assert.equal(MAX_COMPARED, 6);
    assert.equal(new Set(RUN_COLORS).size, 6);
    assert.equal(runColor(0), RUN_COLORS[0]);
    assert.notEqual(runSymbol(0), runSymbol(1));
  });

  it("parses tags and reports what isn't key=value", () => {
    assert.deepEqual(parseTags(" team=rag, owner = ana ,, bad, =x"), {
      tags: { team: "rag", owner: "ana" },
      invalid: ["bad", "=x"],
    });
    assert.equal(formatTags({ team: "rag", a: "1" }), "a=1, team=rag");
    assert.equal(formatTags(undefined), "");
  });

  it("offers short metric names first, then parameters and tags", () => {
    const choices = columnChoices({
      metrics: [
        "fused.recall@5",
        "latency.total_ms.p95",
        "recall@5",
        "doc-v1.final.hit@5",
      ],
      params: ["rerank.component"],
      tags: ["team"],
    });
    assert.deepEqual(choices, [
      "metrics.recall@5",
      "metrics.latency.total_ms.p95",
      "metrics.fused.recall@5",
      "metrics.doc-v1.final.hit@5",
      "params.rerank.component",
      "tags.team",
    ]);
    assert.equal(columnLabel("params.rerank.component"), "rerank.component");
    assert.deepEqual(columnChoices(undefined), []);
  });

  it("formats cells and groups rows by a column", () => {
    assert.equal(formatValue(0.12345), "0.123");
    assert.equal(formatValue(8), "8");
    assert.equal(formatValue(null), "–");
    assert.equal(formatValue(false), "false");
    const rows = [
      {
        id: "a",
        values: { "params.rerank.component": "rerank/gateway" },
        score: 0.8,
      },
      {
        id: "b",
        values: { "params.rerank.component": "rerank/none" },
        score: 0.7,
      },
      {
        id: "c",
        values: { "params.rerank.component": "rerank/gateway" },
        score: 0.6,
      },
      { id: "d", values: {}, score: null },
    ];
    const groups = groupRows(
      rows,
      "params.rerank.component",
      (row) => row.score,
    );
    assert.deepEqual(groups[0], {
      key: "rerank/gateway",
      count: 2,
      mean: 0.7,
      best: 0.8,
      ids: ["a", "c"],
    });
    assert.deepEqual(
      groups.map((group) => group.key),
      ["rerank/gateway", "rerank/none", "–"],
    );
    assert.equal(groups[2]?.mean, null);
  });

  it("selects up to six runs, in the order they were picked", () => {
    let selected: string[] = [];
    for (const id of ["a", "b", "c", "d", "e", "f", "g"]) {
      selected = toggleSelection(selected, id);
    }
    assert.deepEqual(selected, ["a", "b", "c", "d", "e", "f"]);
    assert.deepEqual(toggleSelection(selected, "c"), ["a", "b", "d", "e", "f"]);
    assert.equal(headlineRecall({ "hit@3": 1, "recall@3": 0.5 }), 0.5);
    assert.equal(headlineRecall({ "mrr@10": 0.4 }), null);
  });
});

describe("notes", () => {
  it("parse Markdown into blocks without making HTML", () => {
    const blocks = parseMarkdown(
      "# Why\n\nFewer **tokens**, same `recall@5`.\nStill *fine*.\n\n- one\n- two\n\n1. first\n\n```\n<b>code</b>\n```\n<script>x</script>",
    );
    assert.deepEqual(
      blocks.map((block) => block.kind),
      ["heading", "paragraph", "list", "list", "code", "paragraph"],
    );
    assert.deepEqual(blocks[1], {
      kind: "paragraph",
      inline: [
        { kind: "text", text: "Fewer " },
        { kind: "strong", text: "tokens" },
        { kind: "text", text: ", same " },
        { kind: "code", text: "recall@5" },
        { kind: "text", text: ". Still " },
        { kind: "em", text: "fine" },
        { kind: "text", text: "." },
      ],
    });
    assert.equal(blocks[3]?.kind === "list" && blocks[3].ordered, true);
    assert.deepEqual(blocks[4], { kind: "code", text: "<b>code</b>" });
    // Markup stays text.
    assert.deepEqual(blocks[5], {
      kind: "paragraph",
      inline: [{ kind: "text", text: "<script>x</script>" }],
    });
    assert.deepEqual(parseInline("a * b"), [{ kind: "text", text: "a * b" }]);
  });
});

describe("model charts", () => {
  it("draw each model's best run with its interval, best first", () => {
    const option = modelBarsOption(
      [
        { label: "embed-bge", best: { mean: 0.8, low: 0.7, high: 0.9 } },
        { label: "embed-local", best: { mean: 0.7, low: 0.6, high: 0.8 } },
        { label: "no runs", best: null },
      ],
      "recall@5",
    );
    const series = option.series as Series;
    assert.deepEqual(series[0]?.data, [0.8, 0.7]);
    assert.deepEqual(series[1]?.data, [
      [0, 0.7, 0.9],
      [1, 0.6, 0.8],
    ]);
    assert.deepEqual((option.yAxis as { data: string[] }).data, [
      "embed-bge",
      "embed-local",
    ]);
    const drawn = horizontalWhisker(null, {
      value: (dimension: number) => [0, 0.6, 0.8][dimension] ?? 0,
      coord: ([x, y]: [number, number]) => [x * 100, y * 40 + 20],
      size: () => [0, 40],
    }) as { children: unknown[] };
    assert.equal(drawn.children.length, 3);
  });

  it("draw controlled pairs on a diverging scale centred on zero", () => {
    const option = pairMatrixOption(
      ["a", "b"],
      [
        [null, { delta: 0.05, pairs: 2 }],
        [{ delta: -0.05, pairs: 2 }, null],
      ],
      "recall@5",
    );
    const visual = option.visualMap as {
      min: number;
      max: number;
      inRange: { color: string[] };
    };
    assert.equal(visual.min, -visual.max);
    assert.equal(visual.max, 0.05);
    assert.deepEqual(visual.inRange.color, DIVERGING);
    // The midpoint is a neutral grey.
    assert.equal(DIVERGING[2], "#B8B8B8");
    const series = option.series as { data: { value: number[] }[] }[];
    assert.deepEqual(
      series[0]?.data.map((cell) => cell.value),
      [
        [1, 0, 0.05],
        [0, 1, -0.05],
      ],
    );
  });
});
