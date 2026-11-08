// Pure helpers for the step inspectors.

export interface Band {
  start: number;
  end: number;
  /** The index of the chunk this stretch of text starts, or null outside every chunk. */
  chunk: number | null;
}

/**
 * Split a document's text into bands at chunk boundaries, for the Chunk
 * inspector's alternating bands. Each band runs from a chunk's start to the
 * next chunk's start (overlap goes to the later chunk) or its own end; text
 * no chunk covers gets a null band. Bands cover [0, length) in order.
 */
export function chunkBands(
  length: number,
  chunks: { char_start: number; char_end: number }[],
): Band[] {
  const clamp = (value: number, low: number) =>
    Math.min(Math.max(value, low), length);
  const order = chunks
    .map((chunk, index) => ({
      start: chunk.char_start,
      end: chunk.char_end,
      index,
    }))
    .sort((a, b) => a.start - b.start || a.index - b.index);
  const bands: Band[] = [];
  let cursor = 0;
  order.forEach((chunk, position) => {
    const start = clamp(chunk.start, cursor);
    if (start > cursor) bands.push({ start: cursor, end: start, chunk: null });
    const next = order[position + 1]?.start ?? Number.POSITIVE_INFINITY;
    const end = clamp(Math.min(chunk.end, next), start);
    if (end > start) bands.push({ start, end, chunk: chunk.index });
    cursor = Math.max(cursor, end);
  });
  if (cursor < length) bands.push({ start: cursor, end: length, chunk: null });
  return bands;
}

/** Category counts as "ssn: 2, card: 1"; counts only, never values. */
export function formatFindings(
  findings: Record<string, unknown> | undefined,
): string {
  const entries = Object.entries(findings ?? {}).filter(
    ([, count]) => typeof count === "number" && count > 0,
  );
  return entries.length
    ? entries.map(([name, count]) => `${name}: ${count}`).join(", ")
    : "–";
}

/** How many of each value a key takes across summaries: the scan outcome summary. */
export function tally(
  summaries: (Record<string, unknown> | null | undefined)[],
  key: string,
): [string, number][] {
  const counts = new Map<string, number>();
  for (const summary of summaries) {
    const value = summary?.[key];
    if (typeof value === "string")
      counts.set(value, (counts.get(value) ?? 0) + 1);
  }
  return [...counts.entries()].sort(
    (a, b) => b[1] - a[1] || a[0].localeCompare(b[0]),
  );
}

/** Extensions whose files unpack into members at intake (unpack/zip, unpack/email). */
export const CONTAINER_EXTENSIONS = [".zip", ".eml", ".msg"];

export interface TreeNode {
  /** The node's own segment: a folder, a container (archive or message) or a file. */
  label: string;
  /** The full entry name, for leaves; null for folders and containers. */
  entry: string | null;
  kind: "folder" | "container" | "file";
  children: TreeNode[];
}

/**
 * The archive and attachment tree of a corpus's entries. Members of an
 * archive or message are named "<container>/<member>" at intake, so a path
 * segment with a container extension that has more segments after it is the
 * container, and everything below it is its members. Children keep the
 * entries' order.
 */
export function entryTree(names: string[]): TreeNode[] {
  const root: TreeNode = {
    label: "",
    entry: null,
    kind: "folder",
    children: [],
  };
  for (const name of names) {
    const parts = name.split("/");
    let node = root;
    parts.forEach((part, index) => {
      const leaf = index === parts.length - 1;
      if (leaf) {
        node.children.push({
          label: part,
          entry: name,
          kind: "file",
          children: [],
        });
        return;
      }
      const container = CONTAINER_EXTENSIONS.some((ext) =>
        part.toLowerCase().endsWith(ext),
      );
      let next = node.children.find(
        (child) => child.entry === null && child.label === part,
      );
      if (!next) {
        next = {
          label: part,
          entry: null,
          kind: container ? "container" : "folder",
          children: [],
        };
        node.children.push(next);
      }
      node = next;
    });
  }
  return root.children;
}

/** Names that share bytes, by SHA-256: groups of two or more, each sorted, ordered by first name. */
export function duplicateGroups(
  entries: { name: string; sha256: string }[],
): string[][] {
  const groups = new Map<string, string[]>();
  for (const entry of entries) {
    groups.set(entry.sha256, [...(groups.get(entry.sha256) ?? []), entry.name]);
  }
  return [...groups.values()]
    .filter((names) => names.length > 1)
    .map((names) => [...names].sort())
    .sort((a, b) => a[0]!.localeCompare(b[0]!));
}

export interface CoverageBand {
  start: number;
  end: number;
  /** The first chunk covering this stretch, or null when none does. */
  chunk: number | null;
  /** More than one chunk covers it: an overlap. */
  overlap: boolean;
}

/**
 * Bands of a text by how many chunks cover each stretch: alternating chunk
 * bands, with overlaps marked so the inspector can hatch them.
 */
export function coverageBands(
  length: number,
  chunks: { char_start: number; char_end: number }[],
): CoverageBand[] {
  const cuts = new Set<number>([0, length]);
  for (const chunk of chunks) {
    cuts.add(Math.max(0, Math.min(chunk.char_start, length)));
    cuts.add(Math.max(0, Math.min(chunk.char_end, length)));
  }
  const points = [...cuts].sort((a, b) => a - b);
  const bands: CoverageBand[] = [];
  for (let index = 0; index + 1 < points.length; index += 1) {
    const start = points[index]!;
    const end = points[index + 1]!;
    if (end <= start) continue;
    const covering = chunks
      .map((chunk, position) => ({ chunk, position }))
      .filter(
        ({ chunk }) => chunk.char_start <= start && chunk.char_end >= end,
      );
    bands.push({
      start,
      end,
      chunk: covering.length ? covering[0]!.position : null,
      overlap: covering.length > 1,
    });
  }
  return bands;
}

export interface HistogramBin {
  from: number;
  to: number;
  count: number;
}

/** Token counts in `bins` equal-width bins from 0 to the largest count (or the limit). */
export function histogram(
  counts: number[],
  bins = 20,
  limit?: number,
): HistogramBin[] {
  const top = Math.max(1, ...counts, limit ?? 0);
  const width = Math.max(1, Math.ceil(top / bins));
  const result: HistogramBin[] = Array.from(
    { length: Math.ceil(top / width) + (top % width === 0 ? 1 : 0) },
    (_, index) => ({ from: index * width, to: (index + 1) * width, count: 0 }),
  );
  for (const count of counts) {
    const bin = result[Math.min(result.length - 1, Math.floor(count / width))]!;
    bin.count += 1;
  }
  return result;
}

export interface ChunkWarning {
  kind: "over_limit" | "tiny" | "split_table";
  chunk: number | null;
  detail: string;
}

/**
 * What the Chunk screen warns about: chunks over the embedding model's input
 * limit (silently truncated), chunks under the tiny threshold, and tables no
 * single chunk holds whole (unless the chunker repeated their header).
 */
export function chunkWarnings(
  chunks: {
    ordinal: number;
    tokens: Record<string, number>;
    char_start: number;
    char_end: number;
    metadata?: Record<string, unknown>;
  }[],
  elements: {
    id: string;
    type: string;
    char_start: number;
    char_end: number;
  }[],
  limit: number,
  tiny: number,
): ChunkWarning[] {
  const warnings: ChunkWarning[] = [];
  for (const chunk of chunks) {
    const tokens = Object.values(chunk.tokens)[0] ?? 0;
    if (tokens > limit)
      warnings.push({
        kind: "over_limit",
        chunk: chunk.ordinal,
        detail: `${tokens} tokens, over ${limit}`,
      });
    else if (tokens < tiny)
      warnings.push({
        kind: "tiny",
        chunk: chunk.ordinal,
        detail: `${tokens} tokens, under ${tiny}`,
      });
  }
  for (const element of elements) {
    if (element.type !== "table") continue;
    const whole = chunks.some(
      (chunk) =>
        chunk.char_start <= element.char_start &&
        chunk.char_end >= element.char_end,
    );
    const repeated = chunks.some(
      (chunk) =>
        chunk.metadata?.table === element.id && chunk.metadata?.header_repeated,
    );
    if (!whole && !repeated)
      warnings.push({
        kind: "split_table",
        chunk: null,
        detail: `table ${element.id} is split across chunks`,
      });
  }
  return warnings;
}
