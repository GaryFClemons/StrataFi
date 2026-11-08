import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  appendItem,
  moveItem,
  removeItem,
  STEPS,
  getAt,
  reaches,
  refAt,
  sameDocument,
  sampleFor,
  setAt,
  slotsFor,
  stableStringify,
  withComponent,
  withParam,
  type SpecDocument,
} from "../src/lab/spec/editing";

// The baseline as POST /pipeline-specs/validate resolves an empty spec.
function baseline(): SpecDocument {
  return {
    schema: "stratafi.pipeline.v1",
    name: "lab",
    description: "",
    stages: {
      extract: {
        component: { component: "extract/upload", params: {} },
        unpack: [{ component: "unpack/zip", params: { max_entries: 10000 } }],
      },
      scan: {
        checks: [
          { component: "scan/content-safety", params: { max_pages: 2000 } },
          { component: "scan/clamav", params: { max_signature_age_hours: 72 } },
          { component: "scan/dedupe", params: {} },
        ],
        on_outcome: { infected: "exclude" },
      },
      parse: {
        route: {
          default: { component: "parse/markdown", params: {} },
        },
      },
      clean: [
        { component: "clean/whitespace", params: { paragraphs: "keep" } },
      ],
      chunk: {
        component: "chunk/sections",
        params: { max_tokens: 800, min_tokens: 500 },
        index_text: "{title}\n{heading_path}\n{text}",
      },
      enrich: { component: "enrich/none", params: {} },
      embed: {
        component: "embed/gateway",
        params: { alias: "embed-default", batch_size: 32 },
      },
      index: {
        dense: { component: "index/exact", params: { metric: "cosine" } },
        keyword: { component: "index/bm25", params: { k1: 1.2, b: 0.75 } },
      },
    },
  };
}

describe("the steps", () => {
  it("run to the last stage each shows; search doesn't run", () => {
    assert.deepEqual(
      STEPS.map((step) => [step.id, step.until]),
      [
        ["extract", "extract"],
        ["scan", "clean"],
        ["parse", "parse"],
        ["chunk", "enrich"],
        ["embed", "index"],
        ["search", null],
      ],
    );
  });

  it("know which runs reached a stage", () => {
    assert.ok(reaches("parse", "scan"));
    assert.ok(reaches("parse", "parse"));
    assert.ok(!reaches("parse", "chunk"));
    assert.ok(reaches(null, "index"));
    assert.ok(!reaches("bogus", "extract"));
  });
});

describe("slots", () => {
  it("list each component a step configures, from the document", () => {
    const document = baseline();
    assert.deepEqual(
      slotsFor("extract", document).map((slot) => slot.path),
      ["stages.extract.component", "stages.extract.unpack.0"],
    );
    assert.deepEqual(
      slotsFor("scan", document).map((slot) => [slot.path, slot.stage]),
      [
        ["stages.scan.checks.0", "scan"],
        ["stages.scan.checks.1", "scan"],
        ["stages.scan.checks.2", "scan"],
      ],
    );
    assert.deepEqual(
      slotsFor("parse", document).map((slot) => slot.label),
      [],
    );
    assert.deepEqual(
      slotsFor("embed", document).map((slot) => [
        slot.path,
        slot.fixed ?? false,
      ]),
      [
        ["stages.embed", false],
        ["stages.index.dense", false],
        ["stages.index.keyword", false],
      ],
    );
    assert.deepEqual(slotsFor("search", document), []);
  });

  it("skip an index the spec leaves out", () => {
    const document = setAt(baseline(), "stages.index.keyword", undefined);
    assert.deepEqual(
      slotsFor("embed", document).map((slot) => slot.path),
      ["stages.embed", "stages.index.dense"],
    );
  });

  it("keep extractors and unpackers apart", () => {
    const [extractor, unpacker] = slotsFor("extract", baseline());
    assert.equal(extractor?.prefix, "extract/");
    assert.equal(unpacker?.prefix, "unpack/");
  });
});

describe("editing", () => {
  it("reads and writes dotted paths without touching the original", () => {
    const original = baseline();
    const changed = setAt(original, "stages.clean.0.params.paragraphs", "trim");
    assert.equal(getAt(changed, "stages.clean.0.params.paragraphs"), "trim");
    assert.equal(getAt(original, "stages.clean.0.params.paragraphs"), "keep");
    assert.equal(getAt(original, "stages.nothing.here"), undefined);
  });

  it("swaps a component and starts its parameters from the new defaults", () => {
    const document = withComponent(
      baseline(),
      "stages.chunk",
      "chunk/fixed-tokens",
      {
        size: 512,
        overlap: 64,
      },
    );
    assert.deepEqual(refAt(document, "stages.chunk"), {
      component: "chunk/fixed-tokens",
      params: { size: 512, overlap: 64 },
    });
    // What isn't the component's stays: the chunk stage's index text.
    assert.equal(
      getAt(document, "stages.chunk.index_text"),
      "{title}\n{heading_path}\n{text}",
    );
  });

  it("sets one parameter", () => {
    const document = withParam(
      baseline(),
      "stages.embed",
      "alias",
      "embed-local",
    );
    assert.equal(getAt(document, "stages.embed.params.alias"), "embed-local");
    assert.equal(getAt(document, "stages.embed.params.batch_size"), 32);
    assert.throws(() => withParam(baseline(), "stages.nothing", "x", 1));
  });

  it("removes list items and keys", () => {
    const document = setAt(baseline(), "stages.scan.checks.2", undefined);
    assert.equal(
      (getAt(document, "stages.scan.checks") as unknown[]).length,
      2,
    );
  });

  it("compares documents as configurations, whatever the key order", () => {
    const a = { b: 1, a: { d: [1, 2], c: null } };
    const b = { a: { c: null, d: [1, 2] }, b: 1 };
    assert.equal(stableStringify(a), stableStringify(b));
    assert.ok(sameDocument(a, b));
    assert.ok(!sameDocument(a, { ...b, b: 2 }));
    assert.ok(sameDocument({ a: 1, gone: undefined }, { a: 1 }));
  });

  it("samples with a fixed seed, or not at all", () => {
    assert.deepEqual(sampleFor(5), { size: 5, seed: 0 });
    assert.equal(sampleFor(null), null);
  });
});

describe("list editing", () => {
  const document = {
    stages: {
      clean: [
        { component: "clean/whitespace", params: {} },
        { component: "clean/unicode", params: {} },
      ],
    },
  };
  it("moves, removes and appends items", () => {
    const moved = moveItem(document, "stages.clean", 1, 0);
    assert.deepEqual(
      (getAt(moved, "stages.clean") as { component: string }[]).map(
        (rule) => rule.component,
      ),
      ["clean/unicode", "clean/whitespace"],
    );
    assert.equal(moveItem(document, "stages.clean", 0, -1), document);
    const removed = removeItem(document, "stages.clean", 0);
    assert.equal((getAt(removed, "stages.clean") as unknown[]).length, 1);
    const added = appendItem(document, "stages.clean", {
      component: "clean/redact",
      params: { mode: "mask" },
    });
    assert.equal(getAt(added, "stages.clean.2.component"), "clean/redact");
    // The original is untouched.
    assert.equal((getAt(document, "stages.clean") as unknown[]).length, 2);
  });
});
