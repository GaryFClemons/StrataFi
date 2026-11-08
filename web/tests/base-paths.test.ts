// The app works under any base path: the
// standalone /stratafi/, and a host's, such as /a/b/.

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { toBrowserPath, toLabPath } from "../src/lab/paths";
import {
  NAVIGATION,
  STEP_IDS,
  parseRoute,
  routePath,
  type Route,
} from "../src/lab/routes";

const ROUTES: Route[] = [
  { name: "home" },
  { name: "documents" },
  { name: "corpus", corpusId: "c-1" },
  ...STEP_IDS.map((step) => ({ name: "pipelines" as const, step })),
  { name: "runs" },
  { name: "run", runId: "r 1" },
  { name: "search" },
  { name: "chat" },
  { name: "evalsets" },
  { name: "evalset", evalSetId: "e 1" },
  { name: "evals" },
  { name: "eval", evalRunId: "r 1" },
  { name: "eval-query", evalRunId: "r 1", itemId: "q/1" },
  { name: "eval-compare", a: "r 1", b: "r/2" },
];

for (const base of ["/stratafi/", "/a/b/"]) {
  describe(`under ${base}`, () => {
    it("links every route inside the base path and reads it back", () => {
      for (const route of ROUTES) {
        const href = toBrowserPath(base, routePath(route));
        assert.ok(href.startsWith(base), href);
        assert.deepEqual(parseRoute(toLabPath(base, href) ?? "?"), route);
      }
    });

    it("opens Home at the base path, with or without its slash", () => {
      assert.deepEqual(parseRoute(toLabPath(base, base) ?? "?"), {
        name: "home",
      });
      assert.deepEqual(parseRoute(toLabPath(base, base.slice(0, -1)) ?? "?"), {
        name: "home",
      });
    });

    it("leaves paths outside the base path alone", () => {
      assert.equal(toLabPath(base, "/api/v1/runs"), null);
      assert.equal(toLabPath(base, `${base.slice(0, -1)}x/runs`), null);
    });

    it("keeps the navigation inside the base path", () => {
      const hrefs = NAVIGATION.map((item) =>
        toBrowserPath(base, routePath(item.route)),
      );
      assert.deepEqual(
        hrefs,
        [
          "",
          "chat",
          "documents",
          "pipelines/extract",
          "maps",
          "runs",
          "search",
          "eval-sets",
          "evals",
          "models",
          "experiments",
          "training",
          "profiles",
          "tests",
        ].map((path) => base + path),
      );
    });
  });
}
