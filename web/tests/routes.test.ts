import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  NAVIGATION,
  STEP_IDS,
  parseRoute,
  routePath,
  sectionOf,
  type Route,
} from "../src/lab/routes";

const ROUTES: Route[] = [
  { name: "home" },
  { name: "documents" },
  { name: "corpus", corpusId: "7c1e2b2e-0000-4000-8000-000000000001" },
  ...STEP_IDS.map((step) => ({ name: "pipelines" as const, step })),
  { name: "runs" },
  { name: "run", runId: "0b5ee1a4-0000-4000-8000-000000000002" },
  { name: "search" },
  { name: "chat" },
  { name: "evalsets" },
  { name: "evalset", evalSetId: "5e2a9c1d-0000-4000-8000-000000000003" },
  { name: "evals" },
  { name: "eval", evalRunId: "9a1b2c3d-0000-4000-8000-000000000004" },
  { name: "eval-query", evalRunId: "9a1b", itemId: "cg-board--as-staff" },
  { name: "eval-compare", a: null, b: null },
  { name: "eval-compare", a: "r1", b: "r2" },
  { name: "answers", answerRunId: "1d2c3b4a-0000-4000-8000-000000000007" },
  { name: "experiments" },
  { name: "experiment", experimentId: "3f2a9c1d-0000-4000-8000-000000000005" },
  { name: "profiles" },
  { name: "profile", profileId: "5e2a9c1d-0000-4000-8000-000000000006" },
  { name: "tests" },
  { name: "suite", suiteName: "policies-regression" },
  { name: "test-run", testRunId: "6f1b2c4d-0000-4000-8000-000000000008" },
  { name: "automation" },
  { name: "compare", runs: ["r1", "r2", "r3"] },
  { name: "models" },
  { name: "maps" },
  { name: "map", pipeline: "p1", retrieval: "c1" },
  { name: "map", pipeline: "p1", retrieval: null },
  { name: "map", pipeline: null, retrieval: "c1" },
];

describe("routes", () => {
  it("parses every screen's path", () => {
    assert.deepEqual(parseRoute("/"), { name: "home" });
    assert.deepEqual(parseRoute(""), { name: "home" });
    assert.deepEqual(parseRoute("/documents"), { name: "documents" });
    assert.deepEqual(parseRoute("/documents/abc"), {
      name: "corpus",
      corpusId: "abc",
    });
    assert.deepEqual(parseRoute("/pipelines/chunk"), {
      name: "pipelines",
      step: "chunk",
    });
    assert.deepEqual(parseRoute("/runs/r1"), { name: "run", runId: "r1" });
    assert.deepEqual(parseRoute("/search"), { name: "search" });
    assert.deepEqual(parseRoute("/chat"), { name: "chat" });
    assert.deepEqual(parseRoute("/eval-sets"), { name: "evalsets" });
    assert.deepEqual(parseRoute("/eval-sets/e1"), {
      name: "evalset",
      evalSetId: "e1",
    });
    assert.deepEqual(parseRoute("/evals/r9/queries/q1"), {
      name: "eval-query",
      evalRunId: "r9",
      itemId: "q1",
    });
    assert.equal(parseRoute("/evals/r9/queries").name, "not-found");
    assert.deepEqual(parseRoute("/evals/compare/r1/r2"), {
      name: "eval-compare",
      a: "r1",
      b: "r2",
    });
    assert.equal(parseRoute("/evals/compare/r1").name, "not-found");
    assert.deepEqual(parseRoute("/experiments/x1"), {
      name: "experiment",
      experimentId: "x1",
    });
    assert.equal(
      sectionOf({ name: "experiment", experimentId: "x1" }),
      "experiments",
    );
  });

  it("compares one to six runs, and maps a spec, a config or both", () => {
    assert.deepEqual(parseRoute("/compare/a/b"), {
      name: "compare",
      runs: ["a", "b"],
    });
    assert.equal(parseRoute("/compare").name, "not-found");
    assert.equal(parseRoute("/compare/1/2/3/4/5/6/7").name, "not-found");
    assert.equal(
      routePath({ name: "map", pipeline: null, retrieval: "c" }),
      "/maps/-/c",
    );
    assert.equal(parseRoute("/maps/-/-").name, "not-found");
    assert.equal(sectionOf({ name: "compare", runs: ["a"] }), "evals");
    assert.equal(
      sectionOf({ name: "map", pipeline: "p", retrieval: null }),
      "maps",
    );
    assert.deepEqual(
      NAVIGATION.map((item) => item.label),
      ["Home", "Chat", "Documents", "Pipelines", "Maps", "Runs", "Search", "Eval sets",
       "Evals", "Models", "Experiments", "Training", "Profiles", "Tests"],
    ); // prettier-ignore
  });

  it("opens Pipelines at its first step", () => {
    assert.deepEqual(parseRoute("/pipelines"), {
      name: "pipelines",
      step: "extract",
    });
    assert.deepEqual(parseRoute("/pipelines/"), {
      name: "pipelines",
      step: "extract",
    });
  });

  it("ignores query strings, fragments and extra slashes", () => {
    assert.deepEqual(parseRoute("/runs/r1?tab=tasks#top"), {
      name: "run",
      runId: "r1",
    });
    assert.deepEqual(parseRoute("//runs//r1/"), { name: "run", runId: "r1" });
  });

  it("knows what isn't a screen", () => {
    assert.deepEqual(parseRoute("/pipelines/compose"), {
      name: "not-found",
      path: "/pipelines/compose",
    });
    assert.deepEqual(parseRoute("/runs/r1/tasks"), {
      name: "not-found",
      path: "/runs/r1/tasks",
    });
    assert.equal(parseRoute("/search/more").name, "not-found");
    assert.equal(parseRoute("/settings").name, "not-found");
  });

  it("round-trips every route, IDs escaped", () => {
    for (const route of ROUTES) {
      assert.deepEqual(parseRoute(routePath(route)), route);
    }
    const odd: Route = { name: "run", runId: "a b/c?d" };
    assert.equal(routePath(odd), "/runs/a%20b%2Fc%3Fd");
    assert.deepEqual(parseRoute(routePath(odd)), odd);
  });

  it("puts each route in its navigation section", () => {
    assert.equal(sectionOf({ name: "corpus", corpusId: "x" }), "documents");
    assert.equal(sectionOf({ name: "run", runId: "x" }), "runs");
    assert.equal(sectionOf({ name: "evalset", evalSetId: "x" }), "evalsets");
    assert.equal(
      sectionOf({ name: "eval-query", evalRunId: "x", itemId: "y" }),
      "evals",
    );
    assert.equal(sectionOf({ name: "pipelines", step: "embed" }), "pipelines");
    assert.equal(sectionOf({ name: "not-found", path: "/x" }), null);
    for (const item of NAVIGATION) {
      assert.equal(sectionOf(item.route), item.section);
    }
  });
});
