import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  parseRoute,
  routePath,
  sectionOf,
  NAVIGATION,
} from "../src/lab/routes";
import {
  DEFAULT_SET_OPTIONS,
  cardSummary,
  epochsOf,
  epochsOption,
  isTrainingActive,
  metricText,
  negativeReasons,
  setCounts,
  variantName,
  withAdapter,
  withReranker,
} from "../src/lab/training";

const MODEL = "00000000-0000-0000-0000-0000000000c3";

describe("training routes", () => {
  it("round-trips every training route", () => {
    const routes = [
      { name: "training" as const },
      { name: "training-set" as const, trainingSetId: "s-1" },
      { name: "training-run" as const, trainingRunId: "r-1" },
      { name: "trained-model" as const, modelId: MODEL },
    ];
    for (const route of routes) {
      const path = routePath(route);
      assert.deepEqual(parseRoute(path), route);
      assert.equal(sectionOf(route), "training");
    }
    assert.equal(routePath(routes[1]!), "/training/sets/s-1");
    assert.equal(parseRoute("/training/unknown/x").name, "not-found");
    assert.ok(NAVIGATION.some((item) => item.section === "training"));
  });
});

describe("training helpers", () => {
  it("knows which statuses are still running", () => {
    assert.ok(isTrainingActive("running"));
    assert.ok(isTrainingActive("cancelling"));
    assert.ok(!isTrainingActive("succeeded"));
  });

  it("reads a training set's counts and the leak guard", () => {
    const rows = setCounts({
      items: 63,
      records: 63,
      pairs: 52,
      abstentions: 11,
      validation: 13,
      positives: 63,
      positives_missed: 1,
      negatives: 498,
      negatives_overlapping: 260,
      leak_guard: { test_items: 27, near_duplicates: 0, test_overlap: 1 },
    });
    assert.equal(rows.length, 7);
    assert.equal(
      rows.find((row) => row.label === "Records")?.value,
      "63 (52 with evidence, 11 without)",
    );
    assert.match(
      rows.find((row) => row.label === "Leak guard")?.value ?? "",
      /0 dev items dropped .* 27 test queries; 100% of test items/,
    );
    assert.deepEqual(setCounts(null), []);
    assert.deepEqual(
      negativeReasons({
        negatives_by_reason: { abstention: 88, retrieved: 410, avoid: 3 },
      }),
      [
        { reason: "retrieved", count: 410 },
        { reason: "abstention", count: 88 },
        { reason: "avoid", count: 3 },
      ],
    );
  });

  it("charts loss and validation MRR by epoch, with the kept epoch and the base", () => {
    const epochs = epochsOf({
      epochs: [
        {
          epoch: 1,
          train_loss: 2,
          validation_loss: 1,
          "recall@10": 0.6,
          "mrr@10": 0.5,
        },
        {
          epoch: 2,
          train_loss: 1,
          validation_loss: 1,
          "recall@10": 0.7,
          "mrr@10": 0.6,
        },
      ],
    });
    const option = epochsOption(epochs, 2, { "mrr@10": 0.4 }) as {
      xAxis: { data: number[] };
      series: {
        data: (number | null)[];
        markPoint?: unknown;
        markLine?: unknown;
      }[];
    };
    assert.deepEqual(option.xAxis.data, [1, 2]);
    assert.deepEqual(option.series[0]?.data, [2, 1]);
    assert.deepEqual(option.series[1]?.data, [0.5, 0.6]);
    assert.ok(option.series[1]?.markPoint);
    assert.ok(option.series[1]?.markLine);
    assert.deepEqual(epochsOf(null), []);
  });

  it("names variants within the spec's name rule", () => {
    const name = variantName("hybrid-v1", {
      name: "halcyon-adapter",
      version: 2,
    });
    assert.equal(name, "hybrid-v1-halcyon-adapter-v2");
    const long = variantName("x".repeat(80), {
      name: "halcyon-adapter",
      version: 12,
    });
    assert.equal(long.length, 64);
    assert.match(long, /^[a-z0-9][a-z0-9._-]{0,63}$/);
    assert.match(
      variantName("Hybrid Search", { name: "m", version: 1 }),
      /^hybrid-search-m-v1$/,
    );
  });

  it("puts a trained model into a retrieval config", () => {
    const base = {
      schema: "stratafi.retrieval.v1",
      name: "kh",
      retrievers: { dense: { top_n: 40 }, keyword: { top_n: 50 } },
      rerank: { component: "rerank/none" },
    };
    const adapted = withAdapter(base, MODEL, "kh-adapted") as {
      name: string;
      retrievers: { dense: Record<string, unknown> };
    };
    assert.equal(adapted.name, "kh-adapted");
    assert.deepEqual(adapted.retrievers.dense, { top_n: 40, adapter: MODEL });
    assert.deepEqual(base.retrievers.dense, { top_n: 40 }); // the base is untouched
    const reranked = withReranker(base, MODEL, "kh-reranked") as {
      rerank: unknown;
    };
    assert.deepEqual(reranked.rerank, {
      component: "rerank/cross-encoder",
      params: { model: `trained/${MODEL}` },
    });
  });

  it("summarizes a model card", () => {
    const summary = cardSummary({
      use: {
        how: "retrievers.dense.adapter: m",
        needs: ["models.in_process"],
      },
      training: {
        best_epoch: 4,
        best: { "mrr@10": 0.7 },
        baseline: { "mrr@10": 0.5 },
      },
      limits: ["Lab only."],
    });
    assert.equal(summary.how, "retrievers.dense.adapter: m");
    assert.deepEqual(summary.needs, ["models.in_process"]);
    assert.equal(summary.bestEpoch, 4);
    assert.deepEqual(summary.limits, ["Lab only."]);
    assert.equal(cardSummary(null).how, "");
    assert.equal(metricText(0.12345), "0.123");
    assert.equal(metricText(null), "–");
  });

  it("defaults mining's options as the server does", () => {
    assert.equal(DEFAULT_SET_OPTIONS.max_negatives, 8);
    assert.deepEqual(DEFAULT_SET_OPTIONS.checkpoints, [
      "dense",
      "keyword",
      "candidates",
    ]);
  });
});
