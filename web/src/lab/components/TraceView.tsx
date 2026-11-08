// A query's trace: retrieval's steps on a timeline, then each list
// with where its first relevant chunk is. IDs and numbers only; the passages'
// text is the query detail's.

import type { components } from "../api";
import { CHECKPOINT_NAMES } from "../compare";
import { spanColor, waterfallOption } from "../traces";
import { Chart } from "./Chart";
import { DataTable } from "./ui";

type TraceBody = components["schemas"]["TraceBody"];
type SpanOut = components["schemas"]["SpanOut"];
type TraceListOut = components["schemas"]["TraceListOut"];

function attributes(span: SpanOut): string {
  return Object.entries(span.attributes ?? {})
    .map(([key, value]) => `${key} ${String(value)}`)
    .join(", ");
}

function where(list: TraceListOut): string {
  if (list.found === null || list.found === undefined)
    return `${list.length} entries`;
  return list.found
    ? `${list.length} entries; first relevant at ${list.first_relevant_rank}`
    : `${list.length} entries; no relevant chunk`;
}

export function TraceView({ trace }: { trace: TraceBody }) {
  const spans = trace.spans;
  return (
    <div className="grid gap-3" data-testid="trace">
      <Chart
        option={waterfallOption(spans)}
        label={`The search's steps on a timeline, ${trace.total_ms.toFixed(1)} ms in all`}
        height={Math.max(160, 34 * spans.length + 56)}
        testId="trace-waterfall"
        renderer="svg"
      />
      <DataTable<SpanOut>
        label="Spans"
        rows={spans}
        rowKey={(span) => span.id}
        columns={[
          {
            header: "Step",
            cell: (span) => (
              <span className="inline-flex items-center gap-2">
                <span
                  aria-hidden="true"
                  className="inline-block h-2.5 w-2.5 rounded-sm"
                  style={{ background: spanColor(span.kind) }}
                />
                {span.name}
              </span>
            ),
          },
          { header: "Start (ms)", cell: (span) => span.start_ms.toFixed(2) },
          { header: "Took (ms)", cell: (span) => span.duration_ms.toFixed(2) },
          { header: "Settings", cell: attributes },
          {
            header: "Lists",
            cell: (span) =>
              span.lists
                .map(
                  (list) =>
                    `${CHECKPOINT_NAMES[list.checkpoint] ?? list.checkpoint}: ${where(list)}`,
                )
                .join("; "),
          },
        ]}
      />
      {trace.path.some(
        (entry) => entry.found !== null && entry.found !== undefined,
      ) ? (
        <div
          className="flex flex-wrap items-center gap-1 text-xs"
          data-testid="evidence-path"
        >
          <span className="text-muted">Evidence along the path:</span>
          {trace.path.map((entry) => (
            <span
              key={entry.checkpoint}
              className={`rounded-full border px-2 ${
                entry.found ? "border-success" : "border-danger"
              }`}
            >
              {CHECKPOINT_NAMES[entry.checkpoint] ?? entry.checkpoint}{" "}
              {entry.found ? `✓ ${entry.first_relevant_rank}` : "✗"}
            </span>
          ))}
          {trace.lost_at_checkpoint ? (
            <span className="font-semibold">
              Dropped at{" "}
              {CHECKPOINT_NAMES[trace.lost_at_checkpoint] ??
                trace.lost_at_checkpoint}
            </span>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
