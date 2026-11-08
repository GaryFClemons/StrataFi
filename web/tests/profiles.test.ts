import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { configRows } from "../src/lab/profiles";

describe("profile helpers", () => {
  it("lists a configuration's settings by path", () => {
    assert.deepEqual(
      configRows({
        schema: "stratafi.retrieval.v1",
        name: "hybrid",
        retrievers: { dense: { top_n: 50 }, keyword: { top_n: 50 } },
        fusion: { method: "rrf", k: 60 },
        rerank: { component: "rerank/none", params: {} },
        filters: { tags: [], current_only: true },
        stages: { clean: [{ component: "clean/whitespace" }] },
        top_k: 8,
        adapter: null,
      }),
      [
        { path: "retrievers.dense.top_n", value: "50" },
        { path: "retrievers.keyword.top_n", value: "50" },
        { path: "fusion.method", value: "rrf" },
        { path: "fusion.k", value: "60" },
        { path: "rerank.component", value: "rerank/none" },
        { path: "rerank.params", value: "{}" },
        { path: "filters.tags", value: "[]" },
        { path: "filters.current_only", value: "true" },
        { path: "stages.clean.0.component", value: "clean/whitespace" },
        { path: "top_k", value: "8" },
        { path: "adapter", value: "null" },
      ],
    );
    assert.deepEqual(configRows(null), []);
    assert.deepEqual(configRows("text"), []);
  });
});
