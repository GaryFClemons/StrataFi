// Pure helpers for post-training: a training set's
// counts, a training run's epochs as a chart, a model's card, and retrieval
// config variants that use a trained model. Numbers, names and IDs only.

import { PALETTE } from "./charts";

export const ACTIVE_TRAINING_STATUSES = ["queued", "running", "cancelling"];

export function isTrainingActive(status: string): boolean {
  return ACTIVE_TRAINING_STATUSES.includes(status);
}

/** Mining's options, as stratafi_core.training.TrainingSetOptions defaults them. */
export const DEFAULT_SET_OPTIONS = {
  rule: "evidence-coverage-v1" as const,
  max_positives: 3,
  max_negatives: 8,
  checkpoints: ["dense" as const, "keyword" as const, "candidates" as const],
  depth: 20,
  negative_max_grade: 0.3,
  abstentions: true,
  validation_share: 0.2,
  duplicate_threshold: 0.8,
  seed: 0,
};

export const MODEL_KINDS: Record<string, string> = {
  "query-adapter": "Query adapter",
  embedder: "Embedder",
  reranker: "Reranker",
};

export const EXPORT_FORMATS = [
  {
    format: "sentence-transformers",
    label: "Embedder pairs",
    help: "anchor, positive, negatives",
  },
  {
    format: "cross-encoder",
    label: "Reranker pairs",
    help: "query, passage, label",
  },
  {
    format: "chat",
    label: "Grounded QA chat",
    help: "for fine-tuning a chat model",
  },
] as const;

export interface CountRow {
  label: string;
  value: string;
}

function count(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

/** A training set's headline counts, in reading order. */
export function setCounts(
  stats: Record<string, unknown> | null | undefined,
): CountRow[] {
  if (!stats) return [];
  const leak = (stats.leak_guard ?? {}) as Record<string, unknown>;
  const overlap = leak.test_overlap;
  return [
    { label: "Items read", value: String(count(stats.items)) },
    {
      label: "Records",
      value: `${count(stats.records)} (${count(stats.pairs)} with evidence, ${count(stats.abstentions)} without)`,
    },
    {
      label: "Held out for validation",
      value: String(count(stats.validation)),
    },
    {
      label: "Positives",
      value: `${count(stats.positives)} (${count(stats.positives_missed)} that retrieval missed)`,
    },
    { label: "Hard negatives", value: String(count(stats.negatives)) },
    {
      label: "Left out as near-positives",
      value: String(count(stats.negatives_overlapping)),
    },
    {
      label: "Leak guard",
      value:
        `${count(leak.near_duplicates)} dev items dropped as near duplicates of ` +
        `${count(leak.test_items)} test queries` +
        (typeof overlap === "number"
          ? `; ${Math.round(overlap * 100)}% of test items share a document with the positives`
          : ""),
    },
  ];
}

/** Negatives by reason, most first. */
export function negativeReasons(
  stats: Record<string, unknown> | null | undefined,
): { reason: string; count: number }[] {
  const reasons = (stats?.negatives_by_reason ?? {}) as Record<string, unknown>;
  return Object.entries(reasons)
    .map(([reason, value]) => ({ reason, count: count(value) }))
    .sort((a, b) => b.count - a.count || a.reason.localeCompare(b.reason));
}

export interface Epoch {
  epoch: number;
  train_loss: number | null;
  validation_loss: number | null;
  "recall@10": number | null;
  "mrr@10": number | null;
}

/** A run's epochs from its metrics (or its progress while it runs). */
export function epochsOf(
  metrics: Record<string, unknown> | null | undefined,
): Epoch[] {
  const epochs = metrics?.epochs;
  return Array.isArray(epochs) ? (epochs as Epoch[]) : [];
}

/** Training loss and validation MRR@10 by epoch, the epoch kept marked. */
export function epochsOption(
  epochs: Epoch[],
  bestEpoch: number | null | undefined,
  baseline?: { "mrr@10"?: number | null } | null,
): Record<string, unknown> {
  const text = "currentColor";
  const muted = "#8a8a8a";
  const axis = (name: string, extra: Record<string, unknown> = {}) => ({
    name,
    nameTextStyle: { color: text },
    axisLabel: { color: text },
    axisLine: { lineStyle: { color: muted } },
    splitLine: { lineStyle: { color: muted, opacity: 0.25 } },
    ...extra,
  });
  const base = baseline?.["mrr@10"];
  return {
    animation: false,
    aria: { enabled: true },
    color: [PALETTE[0], PALETTE[1]],
    tooltip: { trigger: "axis" },
    legend: { top: 0, left: "center", textStyle: { color: text } },
    grid: { left: 56, right: 56, top: 56, bottom: 48 },
    xAxis: axis("Epoch", {
      type: "category",
      nameLocation: "middle",
      nameGap: 28,
      data: epochs.map((epoch) => epoch.epoch),
    }),
    yAxis: [
      axis("Training loss", { type: "value", scale: true }),
      axis("Validation MRR@10", { type: "value", min: 0, max: 1 }),
    ],
    series: [
      {
        name: "Training loss",
        type: "line",
        data: epochs.map((epoch) => epoch.train_loss),
      },
      {
        name: "Validation MRR@10",
        type: "line",
        yAxisIndex: 1,
        data: epochs.map((epoch) => epoch["mrr@10"]),
        markPoint:
          bestEpoch && bestEpoch > 0
            ? {
                data: [
                  {
                    name: "Kept",
                    coord: [
                      String(bestEpoch),
                      epochs.find((epoch) => epoch.epoch === bestEpoch)?.[
                        "mrr@10"
                      ] ?? 0,
                    ],
                  },
                ],
              }
            : undefined,
        markLine:
          typeof base === "number"
            ? {
                symbol: "none",
                label: { formatter: "Base", color: text },
                lineStyle: { type: "dashed", color: muted },
                data: [{ yAxis: base }],
              }
            : undefined,
      },
    ],
  };
}

/** A valid spec name for a variant: lowercase, at most 64 characters. */
export function variantName(
  base: string,
  model: { name: string; version: number },
): string {
  const suffix = `-${model.name}-v${model.version}`;
  const stem = base
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^[^a-z0-9]+/, "");
  return `${stem.slice(0, Math.max(1, 64 - suffix.length))}${suffix}`.slice(
    0,
    64,
  );
}

type Document = Record<string, unknown>;

/** A retrieval config that searches through a trained query adapter. */
export function withAdapter(
  document: Document,
  modelId: string,
  name: string,
): Document {
  const retrievers = {
    ...((document.retrievers ?? {}) as Record<string, unknown>),
  };
  const dense = (retrievers.dense ?? {}) as Record<string, unknown>;
  retrievers.dense = { top_n: 50, ...dense, adapter: modelId };
  return { ...document, name, retrievers };
}

/** A retrieval config that reranks with a trained cross-encoder. */
export function withReranker(
  document: Document,
  modelId: string,
  name: string,
): Document {
  return {
    ...document,
    name,
    rerank: {
      component: "rerank/cross-encoder",
      params: { model: `trained/${modelId}` },
    },
  };
}

export interface CardSummary {
  how: string;
  needs: string[];
  limits: string[];
  bestEpoch: number | null;
  best: Record<string, unknown> | null;
  baseline: Record<string, unknown> | null;
}

/** What a model card says about using the model. */
export function cardSummary(
  card: Record<string, unknown> | null | undefined,
): CardSummary {
  const use = (card?.use ?? {}) as Record<string, unknown>;
  const training = (card?.training ?? {}) as Record<string, unknown>;
  return {
    how: typeof use.how === "string" ? use.how : "",
    needs: Array.isArray(use.needs) ? (use.needs as string[]) : [],
    limits: Array.isArray(card?.limits) ? (card.limits as string[]) : [],
    bestEpoch:
      typeof training.best_epoch === "number" ? training.best_epoch : null,
    best: (training.best as Record<string, unknown> | null) ?? null,
    baseline: (training.baseline as Record<string, unknown> | null) ?? null,
  };
}

/** A number from a metrics row, as text. */
export function metricText(value: unknown, digits = 3): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "–";
}
