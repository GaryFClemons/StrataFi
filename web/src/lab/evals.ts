// Pure helpers for the eval screens:
// numbers with their intervals, metric lookups, and evidence highlights split
// into plain-text segments (text is shown as text, never as markup).

export const RULES = [
  "knowledge-hit-v1",
  "evidence-coverage-v1",
  "doc-v1",
] as const;
export type Rule = (typeof RULES)[number];

export const CHECKPOINTS = [
  "dense",
  "keyword",
  "fused",
  "candidates",
  "reranked",
  "final",
] as const;
export type Checkpoint = (typeof CHECKPOINTS)[number];

export const CHECKPOINT_LABELS: Record<string, string> = {
  dense: "Dense",
  keyword: "Keyword (BM25)",
  fused: "Fused",
  candidates: "Candidates (to the reranker)",
  reranked: "Reranked",
  final: "Final (top k)",
};

export const LOST_LABELS: Record<string, string> = {
  parse: "Parse: the evidence isn't in its document's text",
  retrieval: "Retrieval: no relevant chunk reached the candidates",
  rerank: "Rerank: a relevant candidate was cut from the top k",
};

export interface Estimate {
  mean: number;
  low: number;
  high: number;
  n: number;
}

function isEstimate(value: unknown): value is Estimate {
  return (
    value !== null &&
    typeof value === "object" &&
    typeof (value as Estimate).mean === "number" &&
    typeof (value as Estimate).low === "number" &&
    typeof (value as Estimate).high === "number"
  );
}

/** metrics.metrics[rule][checkpoint][name], if it's there. */
export function estimateOf(
  metrics: Record<string, unknown> | null | undefined,
  rule: string,
  checkpoint: string,
  name: string,
): Estimate | null {
  const byRule = (metrics?.metrics as Record<string, unknown> | undefined)?.[
    rule
  ];
  const byCheckpoint = (byRule as Record<string, unknown> | undefined)?.[
    checkpoint
  ];
  const value = (byCheckpoint as Record<string, unknown> | undefined)?.[name];
  return isEstimate(value) ? value : null;
}

/** "0.923 (0.846–0.981)", or "–". */
export function formatEstimate(
  estimate: Estimate | null | undefined,
  digits = 3,
): string {
  if (!estimate) return "–";
  const fixed = (value: number) => value.toFixed(digits);
  return `${fixed(estimate.mean)} (${fixed(estimate.low)}–${fixed(estimate.high)})`;
}

/** A per-query metric value: score.metrics[rule][checkpoint][name]. */
export function queryValue(
  score: Record<string, unknown> | null | undefined,
  rule: string,
  checkpoint: string,
  name: string,
): number | null {
  const metrics = score?.metrics as
    Record<string, Record<string, Record<string, number>>> | undefined;
  const value = metrics?.[rule]?.[checkpoint]?.[name];
  return typeof value === "number" ? value : null;
}

export interface Segment {
  text: string;
  marked: boolean;
}

/**
 * Text cut into plain and marked segments by highlight spans ([start, end)
 * offsets). Overlapping and out-of-range spans are merged and clipped.
 */
export function highlightSegments(
  text: string,
  spans: readonly (readonly [number, number])[],
): Segment[] {
  const clipped = spans
    .map(([start, end]) => [
      Math.max(0, Math.min(start, text.length)),
      Math.max(0, Math.min(end, text.length)),
    ])
    .filter(([start, end]) => (end ?? 0) > (start ?? 0))
    .sort((a, b) => (a[0] ?? 0) - (b[0] ?? 0));
  const merged: [number, number][] = [];
  for (const [start = 0, end = 0] of clipped) {
    const last = merged[merged.length - 1];
    if (last && start <= last[1]) last[1] = Math.max(last[1], end);
    else merged.push([start, end]);
  }
  const segments: Segment[] = [];
  let position = 0;
  for (const [start, end] of merged) {
    if (start > position) {
      segments.push({ text: text.slice(position, start), marked: false });
    }
    segments.push({ text: text.slice(start, end), marked: true });
    position = end;
  }
  if (position < text.length) {
    segments.push({ text: text.slice(position), marked: false });
  }
  return segments;
}
