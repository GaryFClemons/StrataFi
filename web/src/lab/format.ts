// Formatting for tables and summaries. Numbers and names only: document text
// is shown as text by the inspectors, never formatted into messages.

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return "–";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return unit === 0
    ? `${value} B`
    : `${value.toFixed(value < 10 ? 1 : 0)} ${units[unit]}`;
}

export function formatDuration(
  milliseconds: number | null | undefined,
): string {
  if (milliseconds === null || milliseconds === undefined) return "–";
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  const seconds = milliseconds / 1000;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes} min ${Math.round(seconds % 60)} s`;
}

export function formatTime(iso: string | null | undefined): string {
  if (!iso) return "–";
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? "–" : date.toLocaleString();
}

/** The first characters of a hash or key, enough to tell them apart. */
export function shortHash(
  value: string | null | undefined,
  length = 12,
): string {
  return value ? value.slice(0, length) : "–";
}

export const ACTIVE_RUN_STATUSES = ["queued", "running", "cancelling"];

export function isActive(status: string): boolean {
  return ACTIVE_RUN_STATUSES.includes(status);
}

export type Tone = "default" | "accent" | "success" | "warning" | "danger";

/** The colour a run, task or entry status shows in. */
export function toneOf(status: string): Tone {
  switch (status) {
    case "succeeded":
    case "done":
    case "continue":
    case "passed":
    case "pass":
    case "delivered":
      return "success";
    case "cached":
    case "running":
    case "queued":
    case "active":
    case "evaluating":
    case "pending":
      return "accent";
    case "failed":
    case "fail":
    case "error":
    case "exclude":
    case "excluded":
    case "unsupported_type":
      return "danger";
    case "cancelling":
    case "cancelled":
    case "stopping":
    case "stopped":
    case "flag":
    case "skipped":
      return "warning";
    default:
      return "default";
  }
}

/** Tasks finished (done or reused) out of all, from a run's counts. */
export function progress(tasks: Record<string, number> | undefined): {
  finished: number;
  total: number;
} {
  const counts = tasks ?? {};
  const total = Object.values(counts).reduce((sum, count) => sum + count, 0);
  const finished = [
    "succeeded",
    "cached",
    "failed",
    "cancelled",
    "skipped",
  ].reduce((sum, status) => sum + (counts[status] ?? 0), 0);
  return { finished, total };
}
