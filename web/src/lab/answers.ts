// Pure helpers for answer runs and the question review queue: metric
// rows, labels and agreement. Numbers, names and flags only.

import type { Estimate } from "./evals";

export const ACTIVE_ANSWER_STATUSES = ["queued", "running"];

export function isAnswerRunActive(status: string): boolean {
  return ACTIVE_ANSWER_STATUSES.includes(status);
}

/** The answer metrics, in the order the run shows them. */
export const ANSWER_METRICS: { key: string; label: string; help: string }[] = [
  {
    key: "abstention_accuracy",
    label: "Abstention accuracy",
    help: "Abstains exactly when the corpus holds no evidence",
  },
  {
    key: "answered",
    label: "Answered",
    help: "Queries with evidence that got an answer",
  },
  {
    key: "citation_precision",
    label: "Citation precision",
    help: "Cited sources that hold the evidence",
  },
  {
    key: "citation_recall",
    label: "Citation recall",
    help: "Evidence documents a citation points at",
  },
  {
    key: "faithfulness_pass",
    label: "Faithful",
    help: "The judge scores 4 or 5: every claim in the sources",
  },
  {
    key: "correctness_pass",
    label: "Correct",
    help: "The judge scores 4 or 5 against the reference answer",
  },
  {
    key: "reference_f1",
    label: "Reference F1",
    help: "Word overlap with the reference answer",
  },
];

export interface MetricRow {
  key: string;
  label: string;
  help: string;
  estimate: Estimate | null;
}

function isEstimate(value: unknown): value is Estimate {
  if (value === null || typeof value !== "object") return false;
  const found = value as Record<string, unknown>;
  return ["mean", "low", "high", "n"].every(
    (key) => typeof found[key] === "number",
  );
}

/** A run's metrics as rows; a metric that applies to no query has none. */
export function metricRows(
  metrics: Record<string, unknown> | null | undefined,
): MetricRow[] {
  return ANSWER_METRICS.map((metric) => {
    const value = metrics?.[metric.key];
    return { ...metric, estimate: isEstimate(value) ? value : null };
  });
}

export const LABEL_DIMENSIONS = ["faithful", "correct", "abstained"] as const;
export type Dimension = (typeof LABEL_DIMENSIONS)[number];

export interface Label {
  dimension: string;
  label: boolean;
  labeled_by: string;
}

/** One person's labels on an answer, by dimension. */
export function labelsBy(
  labels: readonly Label[],
  subject: string | null | undefined,
): Partial<Record<Dimension, boolean>> {
  const found: Partial<Record<Dimension, boolean>> = {};
  for (const label of labels) {
    if (label.labeled_by !== subject) continue;
    if ((LABEL_DIMENSIONS as readonly string[]).includes(label.dimension)) {
      found[label.dimension as Dimension] = label.label;
    }
  }
  return found;
}

/** What the judge or the abstention detector said, as a label would. */
export function automaticLabel(
  score: Record<string, unknown>,
  dimension: Dimension,
): boolean | null {
  if (dimension === "abstained") return score.abstained === true;
  const value =
    dimension === "faithful" ? score.faithfulness : score.correctness;
  return typeof value === "number" ? value >= 4 : null;
}

export interface AgreementRow {
  dimension: string;
  n: number;
  agreement: Estimate | null;
  kappa: number | null;
}

export function agreementRows(
  dimensions: Record<string, Record<string, unknown>>,
): AgreementRow[] {
  return LABEL_DIMENSIONS.map((dimension) => {
    const found = dimensions[dimension] ?? {};
    return {
      dimension,
      n: typeof found.n === "number" ? found.n : 0,
      agreement: isEstimate(found.agreement) ? found.agreement : null,
      kappa: typeof found.kappa === "number" ? found.kappa : null,
    };
  });
}

export function formatKappa(kappa: number | null): string {
  return kappa === null ? "–" : kappa.toFixed(2);
}

/** The review queue's candidates by status. */
export function candidateCounts(
  candidates: readonly { status: string }[],
): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const candidate of candidates) {
    counts[candidate.status] = (counts[candidate.status] ?? 0) + 1;
  }
  return counts;
}
