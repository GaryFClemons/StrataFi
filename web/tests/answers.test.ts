import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  agreementRows,
  automaticLabel,
  candidateCounts,
  formatKappa,
  isAnswerRunActive,
  labelsBy,
  metricRows,
} from "../src/lab/answers";
import { parseRoute, routePath, sectionOf } from "../src/lab/routes";

const ESTIMATE = { mean: 0.5, low: 0.25, high: 0.75, n: 12 };

describe("answer helpers", () => {
  it("lists every metric, with or without an estimate", () => {
    const rows = metricRows({
      abstention_accuracy: ESTIMATE,
      reference_f1: null,
      counts: { queries: 12 },
    });
    assert.equal(rows.length, 7);
    assert.deepEqual(rows[0]?.estimate, ESTIMATE);
    assert.equal(
      rows.find((row) => row.key === "reference_f1")?.estimate,
      null,
    );
    assert.deepEqual(
      metricRows(null).map((row) => row.estimate),
      Array(7).fill(null),
    );
  });

  it("reads one person's labels and the automatic ones", () => {
    const labels = [
      { dimension: "faithful", label: true, labeled_by: "u-1" },
      { dimension: "abstained", label: false, labeled_by: "u-1" },
      { dimension: "faithful", label: false, labeled_by: "u-2" },
    ];
    assert.deepEqual(labelsBy(labels, "u-1"), {
      faithful: true,
      abstained: false,
    });
    assert.deepEqual(labelsBy(labels, undefined), {});
    assert.equal(automaticLabel({ faithfulness: 4 }, "faithful"), true);
    assert.equal(automaticLabel({ faithfulness: 3 }, "faithful"), false);
    assert.equal(automaticLabel({ correctness: null }, "correct"), null);
    assert.equal(automaticLabel({ abstained: true }, "abstained"), true);
  });

  it("shows agreement per dimension", () => {
    const rows = agreementRows({
      faithful: { n: 4, agreement: ESTIMATE, kappa: 0.2 },
    });
    assert.deepEqual(
      rows.map((row) => [row.dimension, row.n, row.kappa]),
      [
        ["faithful", 4, 0.2],
        ["correct", 0, null],
        ["abstained", 0, null],
      ],
    );
    assert.equal(formatKappa(0.2), "0.20");
    assert.equal(formatKappa(null), "–");
  });

  it("counts the queue and knows which runs still move", () => {
    assert.deepEqual(
      candidateCounts([
        { status: "proposed" },
        { status: "accepted" },
        { status: "proposed" },
      ]),
      { proposed: 2, accepted: 1 },
    );
    assert.ok(isAnswerRunActive("queued"));
    assert.ok(!isAnswerRunActive("failed"));
  });

  it("routes an answer run under the evals", () => {
    const route = { name: "answers" as const, answerRunId: "a1" };
    assert.equal(routePath(route), "/evals/answers/a1");
    assert.deepEqual(parseRoute("/evals/answers/a1"), route);
    assert.equal(sectionOf(route), "evals");
  });
});
