import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  DEFAULT_CONFIG,
  configOf,
  evidenceOf,
  retrievalDocument,
} from "../src/lab/playground";

describe("playground configs", () => {
  it("become retrieval documents", () => {
    const document = retrievalDocument(
      { ...DEFAULT_CONFIG, rerank: "rerank/fake-overlap", topK: 5 },
      "playground-a",
    );
    assert.deepEqual(document.retrievers, {
      dense: { top_n: 50 },
      keyword: { top_n: 50 },
    });
    assert.deepEqual(document.rerank, {
      component: "rerank/fake-overlap",
      params: {},
    });
    assert.equal(document.top_k, 5);
  });

  it("need no fusion with one retriever, and fusion with two", () => {
    const single = retrievalDocument(
      { ...DEFAULT_CONFIG, keyword: false },
      "a",
    );
    assert.equal((single.fusion as { method: string }).method, "none");
    const both = retrievalDocument({ ...DEFAULT_CONFIG, fusion: "none" }, "b");
    assert.equal((both.fusion as { method: string }).method, "rrf");
  });

  it("round-trip through a saved document", () => {
    const config = {
      ...DEFAULT_CONFIG,
      fusion: "weighted" as const,
      alpha: 0.7,
    };
    assert.deepEqual(configOf(retrievalDocument(config, "x")), config);
  });

  it("read an item's evidence", () => {
    assert.deepEqual(
      evidenceOf([{ evidence: "Call back." }, { section: "Wires" }]),
      ["Call back."],
    );
  });
});
