import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  estimateOf,
  formatEstimate,
  highlightSegments,
  queryValue,
} from "../src/lab/evals";

describe("eval helpers", () => {
  it("formats an estimate with its interval", () => {
    assert.equal(
      formatEstimate({ mean: 0.9230769, low: 0.846, high: 0.98, n: 52 }),
      "0.923 (0.846–0.980)",
    );
    assert.equal(formatEstimate(null), "–");
  });

  it("finds aggregate and per-query metrics, or nothing", () => {
    const metrics = {
      metrics: {
        "knowledge-hit-v1": {
          final: { "hit@5": { mean: 0.5, low: 0.1, high: 0.9, n: 4 } },
        },
      },
    };
    assert.equal(
      estimateOf(metrics, "knowledge-hit-v1", "final", "hit@5")?.mean,
      0.5,
    );
    assert.equal(estimateOf(metrics, "doc-v1", "final", "hit@5"), null);
    assert.equal(estimateOf(null, "doc-v1", "final", "hit@5"), null);
    const score = { metrics: { "doc-v1": { dense: { "mrr@10": 0.25 } } } };
    assert.equal(queryValue(score, "doc-v1", "dense", "mrr@10"), 0.25);
    assert.equal(queryValue(score, "doc-v1", "final", "mrr@10"), null);
  });

  it("splits text into plain and marked segments", () => {
    const text = "An account is new for 30 days.";
    assert.deepEqual(
      highlightSegments(text, [
        [3, 10],
        [8, 13],
        [25, 99],
      ]),
      [
        { text: "An ", marked: false },
        { text: "account is", marked: true },
        { text: " new for 30 ", marked: false },
        { text: "days.", marked: true },
      ],
    );
    assert.deepEqual(highlightSegments(text, []), [{ text, marked: false }]);
    assert.deepEqual(highlightSegments("abc", [[2, 1]]), [
      { text: "abc", marked: false },
    ]);
  });
});
