import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  budgetRows,
  factorPaths,
  halvingRungs,
  isExperimentActive,
  rungTrail,
  sharedShare,
  trialLabel,
  workRows,
} from "../src/lab/experiments";

const PLAN = {
  factors: [
    { path: "chunk", labels: ["a", "b"] },
    { path: "embed.params.alias", labels: ["e1", "e2"] },
  ],
  work: {
    pipelines: 4,
    stages: {
      index: { tasks: 8, unique: 6, cached: 0, to_run: 6 },
      extract: { tasks: 80, unique: 20, cached: 20, to_run: 0 },
      chunk: { tasks: 80, unique: 40, cached: 0, to_run: 40 },
    },
  },
  budget: {
    trials: { planned: 4, max: 60, over: false },
    cost_usd: { estimated: 1.23456, max: 1, over: true },
    hours: { estimated: 0.0123, max: null, over: false },
  },
};

describe("experiment helpers", () => {
  it("lists the unique work in pipeline order", () => {
    assert.deepEqual(
      workRows(PLAN).map((row) => row.stage),
      ["extract", "chunk", "index"],
    );
    assert.deepEqual(workRows(PLAN)[0], {
      stage: "extract",
      tasks: 80,
      unique: 20,
      cached: 20,
      to_run: 0,
    });
    assert.equal(sharedShare(workRows(PLAN)).toFixed(3), "0.607");
    assert.deepEqual(workRows(null), []);
  });

  it("shows the budget check", () => {
    assert.deepEqual(
      budgetRows(PLAN).map((row) => [
        row.label,
        row.planned,
        row.max,
        row.over,
      ]),
      [
        ["Trials", "4", "60", false],
        ["Cost (USD)", "1.23", "1", true],
        ["Hours", "0.012", "no limit", false],
      ],
    );
  });

  it("labels trials in factor order", () => {
    const paths = factorPaths(PLAN);
    assert.deepEqual(paths, ["chunk", "embed.params.alias"]);
    assert.equal(
      trialLabel({ "embed.params.alias": "e2", chunk: "b" }, paths),
      "b · e2",
    );
    assert.equal(trialLabel({ z: "1", a: "2" }, []), "2 · 1");
  });

  it("lists successive halving's rungs and a trial's way through them", () => {
    assert.deepEqual(halvingRungs(PLAN), []);
    const plan = {
      ...PLAN,
      search: {
        eta: 3,
        rungs: [
          { queries: 10, keep: 4 },
          { queries: 63, keep: 4 },
        ],
      },
    };
    assert.deepEqual(halvingRungs(plan), [
      { rung: 1, queries: 10, keep: 4 },
      { rung: 2, queries: 63, keep: 4 },
    ]);
    assert.equal(rungTrail([]), "–");
    assert.equal(
      rungTrail([
        { queries: 10, kept: true },
        { queries: 63, kept: true },
      ]),
      "10 → 63",
    );
    assert.equal(rungTrail([{ queries: 10, kept: false }]), "10 (pruned)");
    assert.equal(rungTrail([{ queries: 10, kept: null }]), "10 (stopped)");
  });

  it("knows which statuses are still moving", () => {
    assert.ok(isExperimentActive("running"));
    assert.ok(isExperimentActive("stopping"));
    assert.ok(!isExperimentActive("planned"));
    assert.ok(!isExperimentActive("stopped"));
  });
});
