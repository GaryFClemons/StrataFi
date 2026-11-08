// Test suites on the Tests screen: the
// editor's completions, where a check's evidence opens, and a suite's trends.
// The server validates a suite (POST /test-suites/validate); completions only
// suggest what stratafi.test-suite.v1 allows at the cursor. The key lists below
// follow contracts/schemas/test-suite.v1.schema.json, and a test holds them to
// it.

import type { Route } from "./routes";

export const EXAMPLE_SUITE = `schema: stratafi.test-suite.v1
name: policies-regression
description: Guards the handbook's answers whenever the corpus or the pipeline changes.
target:
  pipeline: baseline-v1
  retrieval: hybrid-v1
corpus: halcyon@latest
eval_set: {name: halcyon-retrieval, split: dev}
baseline: last_passing
triggers:
  - on: corpus_version_frozen
checks:
  - id: recall-at-5
    metric: recall@5
    min: 0.3
  - id: everything-parsed
    stage: parse
    metric: parse.success_rate
    min: 1.0
  - id: latency
    metric: latency_ms.p95
    max: 5000
`;

export const KEYS: Record<string, string[]> = {
  suite: [
    "schema",
    "name",
    "description",
    "target",
    "corpus",
    "eval_set",
    "baseline",
    "time_zone",
    "triggers",
    "checks",
    "outputs",
    "code",
  ],
  target: ["profile", "pipeline", "retrieval"],
  eval_set: ["name", "version", "split"],
  baseline: ["eval_run", "profile", "pipeline", "retrieval", "last_passing"],
  triggers: ["on", "schedule"],
  checks: [
    "id",
    "description",
    "type",
    "metric",
    "min",
    "max",
    "rule",
    "checkpoint",
    "where",
    "vs_baseline",
    "stage",
    "assert",
    "outcomes",
    "detector",
    "categories",
    "query",
    "asker",
    "as_of",
    "expect",
    "expect_not",
    "allow_change",
    "python",
    "timeout_s",
  ],
  where: ["tag", "file_type"],
  vs_baseline: ["max_drop", "max_relative_drop", "significance"],
  expect: ["in_top_k", "doc_key", "version", "section", "contains"],
  expect_not: ["in_top_k", "doc_key", "version"],
  asker: ["name", "user", "roles", "groups"],
  max: ["added", "removed", "changed", "unparseable", "flagged"],
};

const METRICS = [
  "recall@5",
  "recall@10",
  "hit@5",
  "mrr@10",
  "ndcg@10",
  "precision@5",
  "findability",
  "latency_ms.p50",
  "latency_ms.p95",
  "latency_ms.mean",
  "cost_per_1k_queries_usd",
  "ingest_cost_usd",
  "eval_cost_usd",
  "run_duration_s",
  "eval_duration_s",
  "ingest_tokens",
  "eval_tokens",
  "must_not_return_violations",
  "avoid_violations",
  "extract.documents",
  "scan.flagged",
  "scan.excluded",
  "parse.success_rate",
  "parse.failed",
  "parse.evidence_findability",
  "parse.ocr_pages",
  "clean.redactions",
  "chunk.count",
  "chunk.tokens_max",
  "chunk.share_over_embed_limit",
  "embed.truncated_inputs",
];

export const VALUES: Record<string, string[]> = {
  schema: ["stratafi.test-suite.v1"],
  type: [
    "metric",
    "budget",
    "safety",
    "regression",
    "stage",
    "golden",
    "data_change",
    "revision_guard",
    "custom",
  ],
  metric: METRICS,
  stage: [
    "extract",
    "scan",
    "parse",
    "clean",
    "chunk",
    "enrich",
    "embed",
    "index",
  ],
  assert: ["all_succeeded", "no_findings", "no_outcome"],
  rule: ["knowledge-hit-v1", "evidence-coverage-v1", "doc-v1"],
  checkpoint: ["dense", "keyword", "fused", "candidates", "reranked", "final"],
  on: ["corpus_version_frozen", "target_changed"],
  split: ["dev"],
  detector: ["scan/pii"],
  baseline: ["last_passing"],
  allow_change: ["false", "true"],
};

export interface Completion {
  /** The text the completion replaces: the word at the cursor. */
  from: number;
  to: number;
  options: string[];
}

interface Line {
  indent: number;
  key: string | null;
  bare: boolean; // the key has no value on its line: a mapping or a list follows
}

function lineOf(text: string): Line {
  const match = /^(\s*)(-\s+)?(?:([\w@.-]+):(.*))?/.exec(text);
  const spaces = match?.[1]?.length ?? 0;
  const dash = match?.[2]?.length ?? 0;
  const key = match?.[3] ?? null;
  const rest = (match?.[4] ?? "").trim();
  return { indent: spaces + dash, key, bare: key !== null && rest === "" };
}

/** The key a line at ``indent`` belongs under: the nearest bare key above it. */
function parentOf(lines: string[], indent: number): string {
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const text = lines[index] ?? "";
    if (text.trim() === "" || text.trim().startsWith("#")) continue;
    const line = lineOf(text);
    const dashed = /^\s*-\s/.test(text);
    // An earlier item of the same list: keep looking for the list's key.
    if (line.indent < indent || (dashed && line.indent <= indent)) {
      if (line.bare && line.key !== null && !(dashed && line.indent === indent))
        return line.key;
      if (/^\s*-\s/.test(text)) {
        const listIndent = text.indexOf("-");
        return parentOf(lines.slice(0, index), listIndent + 1);
      }
      if (line.indent === 0) return "suite";
    }
  }
  return "suite";
}

/** What may be written at ``offset``: keys at the start of a line, values after a key. */
export function completionsAt(text: string, offset: number): Completion {
  const start = text.lastIndexOf("\n", offset - 1) + 1;
  const before = text.slice(start, offset);
  const value = /^\s*(?:-\s+)?([\w]+):\s*([\w@./-]*)$/.exec(before);
  if (value) {
    const key = value[1] ?? "";
    const word = value[2] ?? "";
    const options = (VALUES[key] ?? []).filter((option) =>
      option.startsWith(word),
    );
    return { from: offset - word.length, to: offset, options };
  }
  const keyed = /^(\s*)(-\s+)?([\w]*)$/.exec(before);
  if (!keyed) return { from: offset, to: offset, options: [] };
  const word = keyed[3] ?? "";
  const indent = (keyed[1]?.length ?? 0) + (keyed[2]?.length ?? 0);
  const above = text.slice(0, start).split("\n");
  const parent = indent === 0 ? "suite" : parentOf(above, indent);
  const options = (KEYS[parent] ?? []).filter((key) => key.startsWith(word));
  return { from: offset - word.length, to: offset, options };
}

/** Write a completion into the text: ``key: `` for a key, the value itself. */
export function applyCompletion(
  text: string,
  completion: Completion,
  option: string,
): { text: string; cursor: number } {
  const before = text.slice(0, completion.from);
  const line = before.slice(before.lastIndexOf("\n") + 1);
  const isKey = !/:[ \t]*$/.test(line);
  const inserted = isKey ? `${option}: ` : option;
  return {
    text: before + inserted + text.slice(completion.to),
    cursor: before.length + inserted.length,
  };
}

export interface EvidenceLink {
  kind: string;
  id: string;
  run_id?: string | null;
  label?: string | null;
}

/** Where a check's evidence opens in the lab, or null when it has no screen. */
export function evidenceRoute(link: EvidenceLink): Route | null {
  switch (link.kind) {
    case "eval_run":
      return { name: "eval", evalRunId: link.id };
    case "query":
      return link.run_id
        ? { name: "eval-query", evalRunId: link.run_id, itemId: link.id }
        : null;
    case "comparison": {
      const [a, b] = link.id.split("..");
      return a && b ? { name: "eval-compare", a, b } : null;
    }
    case "run":
      return { name: "run", runId: link.id };
    case "stage":
    case "chunk":
    case "document":
      return link.run_id ? { name: "run", runId: link.run_id } : null;
    default:
      return null;
  }
}

export interface HistoryRun {
  test_run_id: string;
  created_at: string;
  status: string;
  checks: Record<string, { result: string; value: number | null }>;
}

/** Each run's share of passing checks, oldest first: a suite's sparkline. */
export function passShares(runs: HistoryRun[]): number[] {
  return runs.map((run) => {
    const results = Object.values(run.checks);
    if (results.length === 0) return 0;
    const passed = results.filter(
      (check) => check.result === "pass" || check.result === "skipped",
    ).length;
    return passed / results.length;
  });
}

/** One check's values across runs, oldest first; null where it had none. */
export function checkTrend(runs: HistoryRun[], checkId: string) {
  return runs.map((run) => ({
    at: run.created_at,
    value: run.checks[checkId]?.value ?? null,
    result: run.checks[checkId]?.result ?? null,
  }));
}

/** The checks that appear in a suite's history, in their first run's order. */
export function checkIds(runs: HistoryRun[]): string[] {
  const seen: string[] = [];
  for (const run of runs)
    for (const id of Object.keys(run.checks))
      if (!seen.includes(id)) seen.push(id);
  return seen;
}

/** A check's threshold as people read it: ">= 0.85", "<= 800", "drop <= 0.02". */
export function thresholdText(threshold: Record<string, unknown>): string {
  const parts: string[] = [];
  const number = (key: string) => String(threshold[key]);
  if ("min" in threshold) parts.push(`>= ${number("min")}`);
  if ("max" in threshold && typeof threshold.max !== "object")
    parts.push(`<= ${number("max")}`);
  if ("max_drop" in threshold) parts.push(`drop <= ${number("max_drop")}`);
  if ("max_relative_drop" in threshold)
    parts.push(`relative drop <= ${number("max_relative_drop")}`);
  if ("significance" in threshold) parts.push(`p <= ${number("significance")}`);
  if ("in_top_k" in threshold) parts.push(`in top ${number("in_top_k")}`);
  if ("not_in_top_k" in threshold)
    parts.push(`not in top ${number("not_in_top_k")}`);
  if ("allow_change" in threshold)
    parts.push(threshold.allow_change ? "change allowed" : "no change");
  if (Array.isArray(threshold.outcomes))
    parts.push(`none of ${threshold.outcomes.join(", ")}`);
  if (Array.isArray(threshold.categories))
    parts.push(`none of ${threshold.categories.join(", ")}`);
  return parts.join(", ");
}

export const SCOPES = ["read", "runs:write", "tests:run"] as const;
