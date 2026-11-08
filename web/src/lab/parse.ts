// Pure helpers for the Parse screen: element overlays on a rendered page,
// elements by page, the colour of each element type, and an eval item's
// evidence from a passage.

export interface ParsedElement {
  id: string;
  type: string;
  text: string;
  level?: number | null;
  page?: number | null;
  bbox?: [number, number, number, number] | null;
  table?: {
    rows: number;
    cols: number;
    cells: { r: number; c: number; text: string; header?: boolean }[];
  } | null;
  char_start: number;
  char_end: number;
}

export interface ParsedDocument {
  document_id: string;
  title?: string | null;
  media_type: string;
  parser: { id: string; version: string };
  pages: { number: number; width?: number | null; height?: number | null }[];
  elements: ParsedElement[];
  text: string;
  warnings: string[];
}

/** One colour per element type, distinguishable without relying on red against green. */
export const TYPE_COLOURS: Record<string, string> = {
  title: "#1f77b4",
  heading: "#1f77b4",
  paragraph: "#7f7f7f",
  list_item: "#17becf",
  table: "#ff7f0e",
  figure: "#9467bd",
  caption: "#e377c2",
  code: "#8c564b",
  page_header: "#bcbd22",
  page_footer: "#bcbd22",
  page_number: "#bcbd22",
  other: "#c7c7c7",
};

export function colourOf(type: string): string {
  return TYPE_COLOURS[type] ?? TYPE_COLOURS.other!;
}

/** A box in points from the page's bottom left, as CSS pixels from the top left. */
export function overlayRect(
  bbox: [number, number, number, number],
  pageHeight: number,
  scale: number,
): { left: number; top: number; width: number; height: number } {
  const [x0, y0, x1, y1] = bbox;
  return {
    left: x0 * scale,
    top: (pageHeight - y1) * scale,
    width: Math.max(1, (x1 - x0) * scale),
    height: Math.max(1, (y1 - y0) * scale),
  };
}

/** The pages a document has: its own list, or the pages its elements name. */
export function pageNumbers(document: ParsedDocument): number[] {
  const pages = new Set(document.pages.map((page) => page.number));
  for (const element of document.elements) {
    if (element.page) pages.add(element.page);
  }
  return [...pages].sort((a, b) => a - b);
}

/** A page's elements; a document without pages shows all of them on "page" 0. */
export function elementsOnPage(
  document: ParsedDocument,
  page: number,
): ParsedElement[] {
  if (pageNumbers(document).length === 0) return document.elements;
  return document.elements.filter((element) => element.page === page);
}

/** The heading an element sits under: the nearest earlier heading's text. */
export function sectionOf(
  elements: ParsedElement[],
  index: number,
): string | null {
  for (let position = index; position >= 0; position -= 1) {
    const element = elements[position]!;
    if (element.type === "heading" && position !== index) return element.text;
  }
  return null;
}

/** A corpus entry's name as an eval set's doc_key: its file stem, lowercased. */
export function docKeyOf(entry: string): string {
  const stem = (entry.split("/").pop() ?? entry).replace(/\.[^.]+$/, "");
  const key = stem
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^[^a-z0-9]+/, "");
  return (key || "document").slice(0, 128);
}

/** The evidence an authored item points at: the passage, trimmed to the schema's limit. */
export function evidenceOf(
  entry: string,
  elements: ParsedElement[],
  index: number,
): { doc_key: string; section: string | null; evidence: string } {
  const text = elements[index]!.text.replace(/\s+/g, " ").trim();
  return {
    doc_key: docKeyOf(entry),
    section: sectionOf(elements, index),
    evidence: text.slice(0, 2000),
  };
}

/** A table element's cells as rows, for the table viewer. */
export function tableRows(
  table: NonNullable<ParsedElement["table"]>,
): { text: string; header: boolean }[][] {
  const rows = Array.from({ length: table.rows }, () =>
    Array.from({ length: table.cols }, () => ({ text: "", header: false })),
  );
  for (const cell of table.cells) {
    const row = rows[cell.r];
    if (row && cell.c < row.length)
      row[cell.c] = { text: cell.text, header: Boolean(cell.header) };
  }
  return rows;
}
