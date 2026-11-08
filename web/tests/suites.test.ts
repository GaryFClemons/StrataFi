import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, it } from "node:test";
import {
  KEYS,
  applyCompletion,
  checkIds,
  checkTrend,
  completionsAt,
  evidenceRoute,
  passShares,
  thresholdText,
  type HistoryRun,
} from "../src/lab/suites";

const SCHEMA = JSON.parse(
  readFileSync(
    new URL(
      "../../contracts/schemas/test-suite.v1.schema.json",
      import.meta.url,
    ),
    "utf8",
  ),
) as {
  properties: Record<string, unknown>;
  $defs: Record<string, { properties?: Record<string, unknown> }>;
};

function at(text: string): ReturnType<typeof completionsAt> {
  const offset = text.indexOf("|");
  return completionsAt(text.replace("|", ""), offset);
}

describe("the suite editor's completions", () => {
  it("follow the contract's JSON Schema", () => {
    const keys = (name: string) =>
      Object.keys(SCHEMA.$defs[name]?.properties ?? {}).sort();
    assert.deepEqual(
      [...KEYS.suite].sort(),
      Object.keys(SCHEMA.properties).sort(),
    );
    assert.deepEqual([...KEYS.target].sort(), keys("TargetRef"));
    assert.deepEqual([...KEYS.eval_set].sort(), keys("EvalSetRef"));
    assert.deepEqual([...KEYS.baseline].sort(), keys("BaselineRef"));
    assert.deepEqual([...KEYS.triggers].sort(), keys("TriggerSpec"));
    assert.deepEqual([...KEYS.where].sort(), keys("Where"));
    assert.deepEqual([...KEYS.vs_baseline].sort(), keys("VsBaseline"));
    assert.deepEqual([...KEYS.expect].sort(), keys("Expect"));
    assert.deepEqual([...KEYS.expect_not].sort(), keys("ExpectNot"));
    assert.deepEqual([...KEYS.asker].sort(), keys("AskerSpec"));
    assert.deepEqual([...KEYS.max].sort(), keys("DataChangeMax"));
    const checks = new Set<string>();
    for (const name of [
      "MetricCheck",
      "RegressionCheck",
      "StageCheck",
      "GoldenCheck",
      "DataChangeCheck",
      "RevisionGuardCheck",
      "CustomCheck",
    ])
      for (const key of keys(name)) checks.add(key);
    assert.deepEqual([...KEYS.checks].sort(), [...checks].sort());
  });

  it("offer the keys the cursor's mapping takes", () => {
    assert.deepEqual(at("sc|").options, ["schema"]);
    assert.deepEqual(at("target:\n  pi|").options, ["pipeline"]);
    assert.deepEqual(at("checks:\n  - id: a\n    me|").options, ["metric"]);
    assert.deepEqual(
      at("checks:\n  - id: a\n    metric: x\n  - |").options,
      KEYS.checks,
    );
    assert.deepEqual(at("checks:\n  - id: g\n    expect:\n      in|").options, [
      "in_top_k",
    ]);
    assert.deepEqual(at("triggers:\n  - |").options, ["on", "schedule"]);
  });

  it("offer values after a key", () => {
    assert.deepEqual(at("checks:\n  - stage: pa|").options, ["parse"]);
    assert.deepEqual(at("    metric: recall@|").options, [
      "recall@5",
      "recall@10",
    ]);
    assert.deepEqual(at("triggers:\n  - on: |").options, [
      "corpus_version_frozen",
      "target_changed",
    ]);
    assert.deepEqual(at("name: |").options, []);
  });

  it("write a key with its colon, and a value as it is", () => {
    const text = "checks:\n  - id: a\n    me";
    const completion = completionsAt(text, text.length);
    const done = applyCompletion(text, completion, "metric");
    assert.equal(done.text, "checks:\n  - id: a\n    metric: ");
    assert.equal(done.cursor, done.text.length);
    const value = "    stage: pa";
    const filled = applyCompletion(
      value,
      completionsAt(value, value.length),
      "parse",
    );
    assert.equal(filled.text, "    stage: parse");
  });
});

describe("a check's evidence", () => {
  it("opens the screen that shows it", () => {
    assert.deepEqual(evidenceRoute({ kind: "eval_run", id: "e1" }), {
      name: "eval",
      evalRunId: "e1",
    });
    assert.deepEqual(
      evidenceRoute({ kind: "query", id: "q-1", run_id: "e1" }),
      { name: "eval-query", evalRunId: "e1", itemId: "q-1" },
    );
    assert.deepEqual(evidenceRoute({ kind: "comparison", id: "a..b" }), {
      name: "eval-compare",
      a: "a",
      b: "b",
    });
    assert.deepEqual(evidenceRoute({ kind: "chunk", id: "k", run_id: "r" }), {
      name: "run",
      runId: "r",
    });
    assert.equal(evidenceRoute({ kind: "query", id: "q" }), null);
  });
});

describe("a suite's history", () => {
  const runs: HistoryRun[] = [
    {
      test_run_id: "t1",
      created_at: "2026-10-31T10:00:00Z",
      status: "passed",
      checks: {
        recall: { result: "pass", value: 0.9 },
        golden: { result: "skipped", value: null },
      },
    },
    {
      test_run_id: "t2",
      created_at: "2026-10-31T11:00:00Z",
      status: "failed",
      checks: {
        recall: { result: "fail", value: 0.8 },
        golden: { result: "pass", value: 2 },
        latency: { result: "pass", value: 120 },
      },
    },
  ];

  it("gives each run's share of passing checks", () => {
    assert.deepEqual(passShares(runs), [1, 2 / 3]);
  });

  it("gives a check's values across runs", () => {
    assert.deepEqual(
      checkTrend(runs, "recall").map((point) => point.value),
      [0.9, 0.8],
    );
    assert.deepEqual(checkIds(runs), ["recall", "golden", "latency"]);
  });

  it("reads thresholds as people do", () => {
    assert.equal(thresholdText({ min: 0.85 }), ">= 0.85");
    assert.equal(
      thresholdText({ max_drop: 0.02, significance: 0.05 }),
      "drop <= 0.02, p <= 0.05",
    );
    assert.equal(thresholdText({ allow_change: false }), "no change");
  });
});
