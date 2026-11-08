import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  formatBytes,
  formatDuration,
  progress,
  toneOf,
} from "../src/lab/format";
import { chunkBands, formatFindings, tally } from "../src/lab/inspect";

function chunk(start: number, end: number) {
  return { char_start: start, char_end: end };
}

describe("chunk bands", () => {
  it("cover the text, in order, with gaps outside chunks", () => {
    assert.deepEqual(chunkBands(20, [chunk(2, 8), chunk(10, 15)]), [
      { start: 0, end: 2, chunk: null },
      { start: 2, end: 8, chunk: 0 },
      { start: 8, end: 10, chunk: null },
      { start: 10, end: 15, chunk: 1 },
      { start: 15, end: 20, chunk: null },
    ]);
  });

  it("give overlap to the later chunk", () => {
    assert.deepEqual(chunkBands(12, [chunk(0, 8), chunk(5, 12)]), [
      { start: 0, end: 5, chunk: 0 },
      { start: 5, end: 12, chunk: 1 },
    ]);
  });

  it("sort chunks and stay inside the text", () => {
    assert.deepEqual(chunkBands(10, [chunk(6, 30), chunk(0, 6)]), [
      { start: 0, end: 6, chunk: 1 },
      { start: 6, end: 10, chunk: 0 },
    ]);
    assert.deepEqual(chunkBands(4, []), [{ start: 0, end: 4, chunk: null }]);
    assert.deepEqual(chunkBands(0, [chunk(0, 0)]), []);
  });
});

describe("summaries", () => {
  it("show finding counts, never values", () => {
    assert.equal(
      formatFindings({ ssn: 2, card: 0, iban: 1 }),
      "ssn: 2, iban: 1",
    );
    assert.equal(formatFindings({}), "–");
    assert.equal(formatFindings(undefined), "–");
  });

  it("tally outcomes, most common first", () => {
    assert.deepEqual(
      tally(
        [
          { outcome: "clean" },
          { outcome: "duplicate" },
          { outcome: "clean" },
          null,
          {},
        ],
        "outcome",
      ),
      [
        ["clean", 2],
        ["duplicate", 1],
      ],
    );
  });

  it("format sizes, durations and progress", () => {
    assert.equal(formatBytes(512), "512 B");
    assert.equal(formatBytes(2048), "2.0 KB");
    assert.equal(formatBytes(5 * 1024 * 1024), "5.0 MB");
    assert.equal(formatBytes(undefined), "–");
    assert.equal(formatDuration(450), "450 ms");
    assert.equal(formatDuration(2500), "2.5 s");
    assert.equal(formatDuration(125_000), "2 min 5 s");
    assert.deepEqual(progress({ succeeded: 3, cached: 2, queued: 1 }), {
      finished: 5,
      total: 6,
    });
    assert.deepEqual(progress(undefined), { finished: 0, total: 0 });
  });

  it("colour statuses", () => {
    assert.equal(toneOf("succeeded"), "success");
    assert.equal(toneOf("excluded"), "danger");
    assert.equal(toneOf("cancelled"), "warning");
    assert.equal(toneOf("cached"), "accent");
    assert.equal(toneOf("whatever"), "default");
  });
});
