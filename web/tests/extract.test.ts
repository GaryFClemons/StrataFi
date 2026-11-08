import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { duplicateGroups, entryTree } from "../src/lab/inspect";

describe("entry tree", () => {
  it("nests archive and message members under their container", () => {
    const tree = entryTree([
      "bundle.zip/policies/fees.md",
      "bundle.zip/wires.pdf",
      "inbox/notice.eml/message.txt",
      "inbox/notice.eml/fees.docx",
      "loose.txt",
    ]);
    assert.deepEqual(
      tree.map((node) => [node.label, node.kind]),
      [
        ["bundle.zip", "container"],
        ["inbox", "folder"],
        ["loose.txt", "file"],
      ],
    );
    const bundle = tree[0]!;
    assert.deepEqual(
      bundle.children.map((node) => [node.label, node.kind, node.entry]),
      [
        ["policies", "folder", null],
        ["wires.pdf", "file", "bundle.zip/wires.pdf"],
      ],
    );
    const message = tree[1]!.children[0]!;
    assert.equal(message.kind, "container");
    assert.deepEqual(
      message.children.map((node) => node.entry),
      ["inbox/notice.eml/message.txt", "inbox/notice.eml/fees.docx"],
    );
  });

  it("keeps a refused archive as a file", () => {
    assert.deepEqual(entryTree(["big.zip"]), [
      { label: "big.zip", entry: "big.zip", kind: "file", children: [] },
    ]);
  });
});

describe("duplicate groups", () => {
  it("groups names that share bytes", () => {
    assert.deepEqual(
      duplicateGroups([
        { name: "b/copy.md", sha256: "1" },
        { name: "a/original.md", sha256: "1" },
        { name: "unique.pdf", sha256: "2" },
        { name: "z.txt", sha256: "3" },
        { name: "y.txt", sha256: "3" },
      ]),
      [
        ["a/original.md", "b/copy.md"],
        ["y.txt", "z.txt"],
      ],
    );
  });
});
