import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  docKeyOf,
  elementsOnPage,
  evidenceOf,
  overlayRect,
  pageNumbers,
  sectionOf,
  tableRows,
  type ParsedDocument,
  type ParsedElement,
} from "../src/lab/parse";

function element(
  id: string,
  type: string,
  text: string,
  page: number | null = 1,
): ParsedElement {
  return { id, type, text, page, char_start: 0, char_end: text.length };
}

const DOCUMENT: ParsedDocument = {
  document_id: "d",
  media_type: "application/pdf",
  parser: { id: "parse/pdfplumber", version: "1.0.0" },
  pages: [
    { number: 1, width: 612, height: 792 },
    { number: 2, width: 612, height: 792 },
  ],
  elements: [
    element("e0", "heading", "Wires"),
    element("e1", "paragraph", "Call   the\nnumber on file."),
    element("e2", "paragraph", "Page two.", 2),
  ],
  text: "",
  warnings: [],
};

describe("parse helpers", () => {
  it("place boxes from the bottom left on a top-left canvas", () => {
    assert.deepEqual(overlayRect([72, 700, 300, 720], 792, 2), {
      left: 144,
      top: 144,
      width: 456,
      height: 40,
    });
  });

  it("list pages and their elements", () => {
    assert.deepEqual(pageNumbers(DOCUMENT), [1, 2]);
    assert.deepEqual(
      elementsOnPage(DOCUMENT, 2).map((item) => item.id),
      ["e2"],
    );
    const flat = {
      ...DOCUMENT,
      pages: [],
      elements: [element("x", "paragraph", "a", null)],
    };
    assert.equal(elementsOnPage(flat, 0).length, 1);
  });

  it("make evidence from a passage", () => {
    assert.equal(sectionOf(DOCUMENT.elements, 1), "Wires");
    assert.equal(sectionOf(DOCUMENT.elements, 0), null);
    assert.equal(docKeyOf("policies/Wire Transfers.pdf"), "wire-transfers");
    assert.deepEqual(evidenceOf("policies/wire.pdf", DOCUMENT.elements, 1), {
      doc_key: "wire",
      section: "Wires",
      evidence: "Call the number on file.",
    });
  });

  it("lay a table's cells out as rows", () => {
    assert.deepEqual(
      tableRows({
        rows: 2,
        cols: 2,
        cells: [
          { r: 0, c: 0, text: "Fee", header: true },
          { r: 0, c: 1, text: "Amount", header: true },
          { r: 1, c: 1, text: "25" },
        ],
      }),
      [
        [
          { text: "Fee", header: true },
          { text: "Amount", header: true },
        ],
        [
          { text: "", header: false },
          { text: "25", header: false },
        ],
      ],
    );
  });
});
