import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { chunkWarnings, coverageBands, histogram } from "../src/lab/inspect";

const chunk = (start: number, end: number) => ({
  char_start: start,
  char_end: end,
});

describe("coverage bands", () => {
  it("mark overlaps and gaps", () => {
    assert.deepEqual(coverageBands(20, [chunk(0, 10), chunk(8, 15)]), [
      { start: 0, end: 8, chunk: 0, overlap: false },
      { start: 8, end: 10, chunk: 0, overlap: true },
      { start: 10, end: 15, chunk: 1, overlap: false },
      { start: 15, end: 20, chunk: null, overlap: false },
    ]);
  });
});

describe("token histogram", () => {
  it("counts into equal bins that reach the limit", () => {
    const bins = histogram([10, 20, 95, 100], 4, 120);
    assert.equal(bins[0]!.from, 0);
    assert.equal(
      bins.reduce((total, bin) => total + bin.count, 0),
      4,
    );
    assert.ok(bins.at(-1)!.to >= 120);
  });
});

describe("chunk warnings", () => {
  it("flag over-limit and tiny chunks, and split tables", () => {
    const chunks = [
      { ordinal: 0, tokens: { "chars/4": 600 }, char_start: 0, char_end: 50 },
      { ordinal: 1, tokens: { "chars/4": 5 }, char_start: 50, char_end: 60 },
      { ordinal: 2, tokens: { "chars/4": 100 }, char_start: 60, char_end: 80 },
    ];
    const elements = [
      { id: "e1", type: "table", char_start: 40, char_end: 70 },
      { id: "e2", type: "table", char_start: 60, char_end: 70 },
    ];
    assert.deepEqual(
      chunkWarnings(chunks, elements, 512, 16).map((w) => [w.kind, w.chunk]),
      [
        ["over_limit", 0],
        ["tiny", 1],
        ["split_table", null],
      ],
    );
    const repeated = chunks.map((item) => ({
      ...item,
      metadata: { table: "e1", header_repeated: true },
    }));
    assert.equal(
      chunkWarnings(repeated, elements.slice(0, 1), 1000, 0).length,
      0,
    );
  });
});
