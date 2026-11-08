// The tracking table's helpers: run colours that follow the run, column
// choices, grouping by a column, tag parsing and selection for Compare. Pure
// functions of the API's records, so they're tested without a browser.

import { PALETTE } from "./charts";

/** A comparison holds up to six runs: the palette's six validated slots. */
export const MAX_COMPARED = 6;
export const RUN_COLORS = PALETTE.slice(0, MAX_COMPARED);
/** Shapes that go with the colours, so a run is never told by colour alone. */
export const RUN_SYMBOLS = [
  "circle",
  "rect",
  "triangle",
  "diamond",
  "pin",
  "roundRect",
];

/** A run's colour: its place in the comparison, never its rank. */
export function runColor(index: number): string {
  return RUN_COLORS[index % RUN_COLORS.length] as string;
}

export function runSymbol(index: number): string {
  return RUN_SYMBOLS[index % RUN_SYMBOLS.length] as string;
}

export const SEARCH_EXAMPLES = [
  "metrics.recall@5 > 0.8",
  "params.rerank.component = 'rerank/none'",
  "tags.team = 'rag' and attributes.split = 'dev'",
  "metrics.latency.total_ms.p95 < 50",
];

export const ORDER_CHOICES = [
  { label: "Newest first", value: "" },
  { label: "recall@5, best first", value: "metrics.recall@5 DESC" },
  { label: "MRR@10, best first", value: "metrics.mrr@10 DESC" },
  { label: "nDCG@10, best first", value: "metrics.ndcg@10 DESC" },
  {
    label: "p95 latency, fastest first",
    value: "metrics.latency.total_ms.p95",
  },
];

const TAG_KEY = /^[A-Za-z0-9_.\-/:]{1,64}$/;

/** "team=rag, owner=ana" → its tags, and the parts that aren't key=value. */
export function parseTags(text: string): {
  tags: Record<string, string>;
  invalid: string[];
} {
  const tags: Record<string, string> = {};
  const invalid: string[] = [];
  for (const part of text.split(",")) {
    const trimmed = part.trim();
    if (!trimmed) continue;
    const at = trimmed.indexOf("=");
    const key = at > 0 ? trimmed.slice(0, at).trim() : "";
    if (!TAG_KEY.test(key)) {
      invalid.push(trimmed);
      continue;
    }
    tags[key] = trimmed.slice(at + 1).trim();
  }
  return { tags, invalid };
}

/** Tags as "key=value, key=value", sorted. */
export function formatTags(tags: Record<string, string> | undefined): string {
  return Object.entries(tags ?? {})
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, value]) => `${key}=${value}`)
    .join(", ");
}

/** Short metric names first (the default rule's final list), then the rest. */
function metricRank(key: string): number {
  if (!key.includes(".")) return 0;
  if (key.startsWith("latency.")) return 1;
  if (/^(dense|keyword|fused|candidates|reranked)\./.test(key)) return 2;
  return 3;
}

/** The columns a person can add to the table: metrics, then parameters, then tags. */
export function columnChoices(
  keys: { metrics: string[]; params: string[]; tags: string[] } | undefined,
  limit = 400,
): string[] {
  if (!keys) return [];
  const metrics = [...keys.metrics].sort(
    (a, b) => metricRank(a) - metricRank(b) || a.localeCompare(b),
  );
  return [
    ...metrics.map((key) => `metrics.${key}`),
    ...keys.params.map((key) => `params.${key}`),
    ...keys.tags.map((key) => `tags.${key}`),
  ].slice(0, limit);
}

/** A column's heading: its key without the entity, which the picker shows. */
export function columnLabel(column: string): string {
  const at = column.indexOf(".");
  return at >= 0 ? column.slice(at + 1) : column;
}

export const MISSING = "–";

/** A cell's text: numbers to three places, missing values as a dash. */
export function formatValue(value: unknown, digits = 3): string {
  if (value === null || value === undefined) return MISSING;
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(digits);
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}

export interface Group {
  key: string;
  count: number;
  mean: number | null;
  best: number | null;
  ids: string[];
}

/** Rows grouped by one column's value, each group's mean and best of a
 * metric; the biggest group first, then by the value. */
export function groupRows<
  T extends { id: string; values: Record<string, unknown> },
>(rows: T[], column: string, metricOf: (row: T) => number | null): Group[] {
  const groups = new Map<string, T[]>();
  for (const row of rows) {
    const key = formatValue(row.values[column]);
    groups.set(key, [...(groups.get(key) ?? []), row]);
  }
  return [...groups.entries()]
    .map(([key, members]) => {
      const scores = members
        .map(metricOf)
        .filter((value): value is number => typeof value === "number");
      return {
        key,
        count: members.length,
        mean: scores.length
          ? scores.reduce((sum, value) => sum + value, 0) / scores.length
          : null,
        best: scores.length ? Math.max(...scores) : null,
        ids: members.map((member) => member.id),
      };
    })
    .sort(
      (a, b) =>
        b.count - a.count ||
        Number(a.key === MISSING) - Number(b.key === MISSING) ||
        a.key.localeCompare(b.key),
    );
}

/** Add a run to the selection (up to six), or take it out. */
export function toggleSelection(selected: string[], id: string): string[] {
  if (selected.includes(id)) return selected.filter((item) => item !== id);
  return selected.length >= MAX_COMPARED ? selected : [...selected, id];
}

/** The headline recall in a tracked run's headline, whatever its k. */
export function headlineRecall(
  headline: Record<string, number | null | undefined>,
): number | null {
  const entry = Object.entries(headline).find(([key]) =>
    key.startsWith("recall@"),
  );
  return typeof entry?.[1] === "number" ? entry[1] : null;
}
