// The tracking table: eval runs searched with the search language,
// MLflow's search_runs, with picked columns, grouping by a column, a chart of
// the matches, and up to six runs selected for the Compare workspace.

import { Button } from "@heroui/react";
import { useMemo, useState, type FormEvent } from "react";
import type { components } from "../api";
import { unwrap, useLoad } from "../api/load";
import { PALETTE } from "../charts";
import { Link, useLab, useNavigate } from "../context";
import { formatTime, shortHash } from "../format";
import {
  MAX_COMPARED,
  ORDER_CHOICES,
  SEARCH_EXAMPLES,
  columnChoices,
  columnLabel,
  formatTags,
  formatValue,
  groupRows,
  headlineRecall,
  toggleSelection,
} from "../tracking";
import { Chart } from "./Chart";
import {
  DataTable,
  Empty,
  ErrorNote,
  Loading,
  Section,
  StatusChip,
  inputClass,
} from "./ui";

type TrackedRunOut = components["schemas"]["TrackedRunOut"];
const LIMIT = 200;
const X_DEFAULT = "metrics.latency.total_ms.p95";
const Y_DEFAULT = "metrics.recall@5";

function number(value: unknown): number | null {
  return typeof value === "number" ? value : null;
}

function scatterOption(rows: TrackedRunOut[], x: string, y: string) {
  const points = rows
    .map((row) => [number(row.values[x]), number(row.values[y]), row.label])
    .filter((point) => point[0] !== null && point[1] !== null);
  return {
    animation: false,
    aria: { enabled: true },
    color: [PALETTE[0]],
    grid: { left: 16, right: 24, top: 24, bottom: 40, containLabel: true },
    tooltip: {
      trigger: "item",
      formatter: (params: { value: [number, number, string] }) =>
        `${params.value[2]}: ${params.value[0]}, ${params.value[1]}`,
    },
    xAxis: {
      type: "value",
      scale: true,
      name: columnLabel(x),
      nameLocation: "middle",
      nameGap: 28,
      nameTextStyle: { color: "currentColor" },
      axisLabel: { color: "currentColor" },
    },
    yAxis: {
      type: "value",
      scale: true,
      name: columnLabel(y),
      nameTextStyle: { color: "currentColor" },
      axisLabel: { color: "currentColor" },
    },
    series: [
      {
        name: "Eval runs",
        type: "scatter",
        symbolSize: 11,
        itemStyle: { borderColor: "#ffffff", borderWidth: 2 },
        data: points,
      },
    ],
  };
}

export function TrackingTable() {
  const { api } = useLab();
  const navigate = useNavigate();
  const [draft, setDraft] = useState("");
  const [filter, setFilter] = useState("");
  const [order, setOrder] = useState("");
  const [archived, setArchived] = useState(false);
  const [columns, setColumns] = useState<string[]>([]);
  const [adding, setAdding] = useState("");
  const [groupBy, setGroupBy] = useState("");
  const [x, setX] = useState(X_DEFAULT);
  const [y, setY] = useState(Y_DEFAULT);
  const [selected, setSelected] = useState<string[]>([]);
  const requested = useMemo(
    () => [...new Set([...columns, x, y, ...(groupBy ? [groupBy] : [])])],
    [columns, x, y, groupBy],
  );
  const found = useLoad(
    async () =>
      unwrap(
        await api.POST("/eval-runs/search", {
          body: {
            filter,
            order_by: order ? [order] : [],
            columns: requested,
            include_archived: archived,
            limit: LIMIT,
          },
        }),
      ),
    JSON.stringify([filter, order, archived, requested]),
  );
  const rows = found.data?.items ?? [];
  const choices = columnChoices(found.data?.keys);
  const groups = groupBy
    ? groupRows(rows, groupBy, (row) => headlineRecall(row.headline))
    : [];

  const search = (event: FormEvent) => {
    event.preventDefault();
    setFilter(draft.trim());
  };

  return (
    <Section
      title="Eval runs"
      description="Search with the search language, MLflow's: metrics., params., tags. and attributes. keys joined by and. Select up to six to compare."
      actions={
        <>
          <Button
            size="sm"
            isDisabled={selected.length < 2}
            onPress={() => navigate({ name: "compare", runs: selected })}
          >
            Compare {selected.length || ""} selected
          </Button>
          <Link
            to={{ name: "eval-compare", a: null, b: null }}
            className="text-link text-sm"
          >
            Compare two runs
          </Link>
        </>
      }
    >
      <div className="grid gap-3">
        <form
          onSubmit={search}
          aria-label="Search eval runs"
          className="grid gap-2"
        >
          <div className="flex flex-wrap items-end gap-2">
            <label className="grid min-w-72 flex-1 gap-1 text-xs font-medium">
              Filter
              <input
                className={`${inputClass} font-mono`}
                value={draft}
                placeholder="metrics.recall@5 > 0.8 and params.rerank.component = 'rerank/none'"
                onChange={(event) => setDraft(event.target.value)}
              />
            </label>
            <label className="grid gap-1 text-xs font-medium">
              Order
              <select
                className={inputClass}
                value={order}
                onChange={(event) => setOrder(event.target.value)}
              >
                {ORDER_CHOICES.map((choice) => (
                  <option key={choice.value} value={choice.value}>
                    {choice.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-1 text-xs font-medium">
              <input
                type="checkbox"
                checked={archived}
                onChange={(event) => setArchived(event.target.checked)}
              />
              Archived too
            </label>
            <Button type="submit" size="sm">
              Search
            </Button>
          </div>
          <div className="flex flex-wrap gap-1 text-xs">
            <span className="text-muted">Try:</span>
            {SEARCH_EXAMPLES.map((example) => (
              <button
                key={example}
                type="button"
                className="rounded-full border border-separator px-2 font-mono hover:bg-default"
                onClick={() => {
                  setDraft(example);
                  setFilter(example);
                }}
              >
                {example}
              </button>
            ))}
          </div>
        </form>
        <ErrorNote error={found.error} />
        <div className="flex flex-wrap items-end gap-2 text-xs">
          <label className="grid gap-1 font-medium">
            Add a column
            <select
              className={inputClass}
              value={adding}
              aria-label="Add a column"
              onChange={(event) => setAdding(event.target.value)}
            >
              <option value="">Choose a metric, parameter or tag</option>
              {choices
                .filter((choice) => !columns.includes(choice))
                .map((choice) => (
                  <option key={choice} value={choice}>
                    {choice}
                  </option>
                ))}
            </select>
          </label>
          <Button
            size="sm"
            variant="secondary"
            isDisabled={!adding}
            onPress={() => {
              setColumns([...columns, adding]);
              setAdding("");
            }}
          >
            Add
          </Button>
          {columns.map((column) => (
            <button
              key={column}
              type="button"
              className="rounded-full border border-separator px-2 font-mono hover:bg-default"
              aria-label={`Remove the column ${column}`}
              onClick={() =>
                setColumns(columns.filter((item) => item !== column))
              }
            >
              {columnLabel(column)} ×
            </button>
          ))}
          <label className="grid gap-1 font-medium">
            Group by
            <select
              className={inputClass}
              value={groupBy}
              onChange={(event) => setGroupBy(event.target.value)}
            >
              <option value="">No grouping</option>
              {choices
                .filter((choice) => !choice.startsWith("metrics."))
                .map((choice) => (
                  <option key={choice} value={choice}>
                    {choice}
                  </option>
                ))}
            </select>
          </label>
        </div>
        {found.data === undefined ? (
          found.error ? null : (
            <Loading />
          )
        ) : rows.length === 0 ? (
          <Empty>No eval runs match.</Empty>
        ) : (
          <>
            <p className="text-xs text-muted" data-testid="search-total">
              {found.data.total} of {found.data.scanned} eval runs match
              {found.data.total > rows.length
                ? `; the first ${rows.length} are shown`
                : ""}
              .
            </p>
            {groupBy ? (
              <DataTable
                label="Groups"
                rows={groups}
                rowKey={(group) => group.key}
                columns={[
                  {
                    header: columnLabel(groupBy),
                    cell: (group) => <code>{group.key}</code>,
                  },
                  { header: "Runs", cell: (group) => group.count },
                  {
                    header: "Mean recall",
                    cell: (group) => formatValue(group.mean),
                  },
                  {
                    header: "Best recall",
                    cell: (group) => formatValue(group.best),
                  },
                ]}
              />
            ) : null}
            <DataTable<TrackedRunOut>
              label="Eval runs"
              rows={rows}
              rowKey={(row) => row.id}
              columns={[
                {
                  header: "Compare",
                  cell: (row) => (
                    <input
                      type="checkbox"
                      aria-label={`Select ${row.label}`}
                      checked={selected.includes(row.id)}
                      disabled={
                        !selected.includes(row.id) &&
                        selected.length >= MAX_COMPARED
                      }
                      onChange={() =>
                        setSelected(toggleSelection(selected, row.id))
                      }
                    />
                  ),
                },
                {
                  header: "Eval run",
                  cell: (row) => (
                    <Link
                      to={{ name: "eval", evalRunId: row.id }}
                      className="text-link"
                    >
                      {row.title ?? row.label}{" "}
                      <code className="text-xs text-muted">
                        {shortHash(row.id, 8)}
                      </code>
                    </Link>
                  ),
                },
                {
                  header: "Status",
                  cell: (row) => <StatusChip status={row.status} />,
                },
                {
                  header: "Eval set",
                  cell: (row) =>
                    `${String(row.attributes.eval_set ?? "–")} v${String(row.attributes.eval_set_version ?? "?")}, ${row.split}`,
                },
                ...["recall", "mrr", "ndcg"].map((metric) => ({
                  header: metric === "recall" ? "recall@k" : `${metric}@10`,
                  cell: (row: TrackedRunOut) =>
                    formatValue(
                      Object.entries(row.headline).find(([key]) =>
                        key.startsWith(`${metric}@`),
                      )?.[1],
                    ),
                })),
                {
                  header: "p95 ms",
                  cell: (row) => formatValue(row.latency_p95_ms, 1),
                },
                { header: "Tags", cell: (row) => formatTags(row.tags) || "–" },
                ...columns.map((column) => ({
                  header: columnLabel(column),
                  cell: (row: TrackedRunOut) => formatValue(row.values[column]),
                })),
                {
                  header: "Started",
                  cell: (row) => formatTime(row.created_at),
                },
              ]}
            />
            <div className="grid gap-2">
              <div className="flex flex-wrap items-end gap-2 text-xs">
                <span className="font-medium">Plot the matches:</span>
                {[["x", x, setX] as const, ["y", y, setY] as const].map(
                  ([axis, value, set]) => (
                    <label key={axis} className="grid gap-1 font-medium">
                      {axis}
                      <select
                        className={inputClass}
                        value={value}
                        onChange={(event) => set(event.target.value)}
                      >
                        {[
                          value,
                          ...choices.filter((choice) => choice !== value),
                        ]
                          .filter(
                            (choice) =>
                              choice.startsWith("metrics.") ||
                              choice.startsWith("params."),
                          )
                          .map((choice) => (
                            <option key={choice} value={choice}>
                              {choice}
                            </option>
                          ))}
                      </select>
                    </label>
                  ),
                )}
              </div>
              <Chart
                option={scatterOption(rows, x, y)}
                label={`${columnLabel(y)} against ${columnLabel(x)} for the matching eval runs`}
                height={260}
                testId="tracking-chart"
                renderer="svg"
              />
            </div>
          </>
        )}
      </div>
    </Section>
  );
}
