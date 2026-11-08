// The Compare workspace: two to six eval runs of one eval
// set version side by side. Each run keeps one colour and shape in every
// chart; every number has its interval; differences are paired and adjusted.

import { useMemo, useState } from "react";
import type { components } from "../api";
import { unwrap, useLoad } from "../api/load";
import { Chart } from "../components/Chart";
import { PipelineMap } from "../components/PipelineMap";
import {
  DataTable,
  Empty,
  ErrorNote,
  Loading,
  Section,
  inputClass,
} from "../components/ui";
import {
  CHECKPOINT_NAMES,
  curvesOption,
  forestOption,
  funnelOption,
  gridOption,
  latencyOption,
  lostOption,
  parallelDimensions,
  parallelOption,
  qualityLatencyOption,
  runLabels,
} from "../compare";
import { Link, useLab, useNavigate } from "../context";
import { CHECKPOINTS, RULES, type Checkpoint, type Rule } from "../evals";
import { shortHash } from "../format";
import { diffMaps } from "../maps";
import { stepBoxOption, type StepProfile } from "../traces";
import { formatTags, formatValue, runColor } from "../tracking";

type CompareManyOut = components["schemas"]["CompareManyOut"];
type CellOut = components["schemas"]["CellOut"];
type MapOut = components["schemas"]["MapOut"];
type PassageOut = components["schemas"]["PassageOut"];

const TABS = [
  "Overview",
  "Charts",
  "Parameters",
  "Queries",
  "Latency",
  "Maps",
] as const;
type Tab = (typeof TABS)[number];

function Swatch({ index }: { index: number }) {
  return (
    <span
      aria-hidden="true"
      className="inline-block h-3 w-3 shrink-0 rounded-sm"
      style={{ background: runColor(index) }}
    />
  );
}

function cellText(cell: CellOut | undefined): string {
  if (!cell || typeof cell.mean !== "number") return "–";
  return `${cell.mean.toFixed(3)} (${formatValue(cell.low)}–${formatValue(cell.high)})`;
}

function deltaText(cell: CellOut | undefined): string {
  if (!cell || typeof cell.delta !== "number") return "baseline";
  const sign = cell.delta >= 0 ? "+" : "";
  const record =
    typeof cell.wins === "number"
      ? ` · ${cell.wins} won, ${cell.losses} lost, ${cell.ties} tied`
      : "";
  return `vs baseline ${sign}${cell.delta.toFixed(3)} (${formatValue(cell.delta_low)}–${formatValue(cell.delta_high)}), p ${formatValue(cell.p_holm)}${record}`;
}

function Overview({
  data,
  labels,
}: {
  data: CompareManyOut;
  labels: string[];
}) {
  return (
    <div className="grid gap-4">
      <Chart
        option={forestOption(data.table, labels)}
        label="Each metric's mean and 95% interval, per run"
        height={Math.max(220, 70 * data.table.length + 80)}
        testId="compare-forest"
        renderer="svg"
      />
      <DataTable
        label="Metrics per run"
        rows={data.table}
        rowKey={(row) => row.metric}
        columns={[
          { header: "Metric", cell: (row) => <code>{row.metric}</code> },
          ...labels.map((label, index) => ({
            header: label,
            cell: (row: (typeof data.table)[number]) => {
              const cell = row.cells[index];
              return (
                <div className="grid gap-0.5" data-testid="compare-cell">
                  <span>{cellText(cell)}</span>
                  <span className="text-xs text-muted">{deltaText(cell)}</span>
                  {cell?.group ? (
                    <span
                      className={`text-xs ${cell.group === "leader" ? "font-semibold" : ""}`}
                      data-group={cell.group}
                    >
                      {cell.group === "leader"
                        ? "best"
                        : cell.group === "tied"
                          ? "tied with the best"
                          : "worse than the best"}
                    </span>
                  ) : null}
                </div>
              );
            },
          })),
        ]}
      />
      <p className="text-xs text-muted">
        {data.paired} queries with evidence, scored by every run, are paired.
        Deltas are against the baseline, with Holm's adjustment across the runs;
        groups are against the best run. Means are over the paired queries.
      </p>
    </div>
  );
}

function Charts({ data, labels }: { data: CompareManyOut; labels: string[] }) {
  const [curve, setCurve] = useState("recall");
  const [slice, setSlice] = useState("file_type");
  const chosen =
    data.curves.find((item) => item.metric === curve) ?? data.curves[0];
  const totalIndex = data.latency.steps.indexOf("total_ms");
  const p95 = data.latency.p95.map((values) =>
    totalIndex >= 0 ? (values[totalIndex] ?? null) : null,
  );
  const quality = data.table[1] ?? data.table[0];
  const slices = data.slices.filter((row) => row.kind === slice);
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="grid gap-2">
        <div className="flex items-center gap-2 text-sm font-medium">
          At every k:
          <select
            aria-label="Curve metric"
            className={inputClass}
            value={chosen?.metric ?? ""}
            onChange={(event) => setCurve(event.target.value)}
          >
            {data.curves.map((item) => (
              <option key={item.metric} value={item.metric}>
                {item.metric}
              </option>
            ))}
          </select>
        </div>
        {chosen ? (
          <Chart
            option={curvesOption(chosen, labels)}
            label={`${chosen.metric} at every k, per run`}
            height={300}
            testId="compare-curves"
            renderer="svg"
          />
        ) : (
          <Empty>No curves.</Empty>
        )}
      </div>
      <div className="grid gap-2">
        <span className="text-sm font-medium">Where the evidence survives</span>
        <Chart
          option={funnelOption(data.funnel, labels)}
          label="The share of queries with evidence in each list, per run"
          height={300}
          testId="compare-funnel"
          renderer="svg"
        />
      </div>
      <div className="grid gap-2">
        <span className="text-sm font-medium">Where queries were lost</span>
        <Chart
          option={lostOption(data.where_lost, labels)}
          label="Queries found and lost by stage, per run"
          height={Math.max(200, 50 * labels.length + 70)}
          testId="compare-lost"
          renderer="svg"
        />
      </div>
      <div className="grid gap-2">
        <span className="text-sm font-medium">Quality against latency</span>
        <Chart
          option={qualityLatencyOption(
            labels.map((label, index) => ({
              label,
              quality: quality?.cells[index]?.mean ?? null,
              latency: p95[index] ?? null,
            })),
            quality?.metric ?? "",
          )}
          label="Each run's quality against its p95 latency, with the frontier"
          height={300}
          testId="compare-pareto"
          renderer="svg"
        />
      </div>
      <div className="grid gap-2 lg:col-span-2">
        <span className="text-sm font-medium">
          Parallel coordinates: the parameters that differ, then the metrics
        </span>
        <Chart
          option={parallelOption(
            parallelDimensions(data.params, data.table, p95),
            labels,
          )}
          label="Parallel coordinates of each run's parameters and metrics"
          height={340}
          testId="compare-parallel"
          renderer="svg"
        />
      </div>
      <div className="grid gap-2 lg:col-span-2">
        <div className="flex items-center gap-2 text-sm font-medium">
          {data.metrics[0]} by
          <select
            aria-label="Slice by"
            className={inputClass}
            value={slice}
            onChange={(event) => setSlice(event.target.value)}
          >
            <option value="file_type">file type</option>
            <option value="tag">tag</option>
          </select>
        </div>
        {slices.length ? (
          <Chart
            option={gridOption(
              slices.map((row) => row.label),
              slices.map((row) => row.values),
              labels,
              data.metrics[0] ?? "",
            )}
            label={`${data.metrics[0]} per ${slice === "tag" ? "tag" : "file type"} and run`}
            height={Math.max(180, 26 * slices.length + 110)}
            testId="compare-slices"
            renderer="svg"
          />
        ) : (
          <Empty>No slices.</Empty>
        )}
      </div>
    </div>
  );
}

function Parameters({
  data,
  labels,
}: {
  data: CompareManyOut;
  labels: string[];
}) {
  if (!data.params.length)
    return <Empty>The runs' parameters are the same.</Empty>;
  return (
    <DataTable
      label="Parameters that differ"
      rows={data.params}
      rowKey={(row) => row.key}
      columns={[
        { header: "Parameter", cell: (row) => <code>{row.key}</code> },
        ...labels.map((label, index) => ({
          header: label,
          cell: (row: (typeof data.params)[number]) => {
            const value = formatValue(row.values[index]);
            const differs = value !== formatValue(row.values[data.baseline]);
            return (
              <span className={differs ? "font-semibold" : "text-muted"}>
                {value}
              </span>
            );
          },
        })),
      ]}
    />
  );
}

function SideBySide({
  ids,
  labels,
  itemId,
}: {
  ids: string[];
  labels: string[];
  itemId: string;
}) {
  const { api } = useLab();
  const details = useLoad(
    () =>
      Promise.all(
        ids.map(async (id) =>
          unwrap(
            await api.GET("/eval-runs/{eval_run_id}/queries/{item_id}", {
              params: { path: { eval_run_id: id, item_id: itemId } },
            }),
          ),
        ),
      ),
    `${ids.join(",")}:${itemId}`,
  );
  if (!details.data)
    return details.error ? <ErrorNote error={details.error} /> : <Loading />;
  return (
    <div
      className="grid gap-3"
      style={{
        gridTemplateColumns: `repeat(${ids.length}, minmax(14rem, 1fr))`,
      }}
    >
      {details.data.map((detail, index) => (
        <div
          key={ids[index]}
          className="grid content-start gap-2"
          data-testid="side-by-side"
        >
          <div className="flex items-center gap-2 text-sm font-semibold">
            <Swatch index={index} />
            {labels[index]}
          </div>
          {(detail.checkpoints.final ?? [])
            .slice(0, 8)
            .map((passage: PassageOut) => (
              <div
                key={passage.chunk_id}
                className={`rounded-md border p-2 text-xs ${
                  passage.relevant["knowledge-hit-v1"]
                    ? "border-success"
                    : "border-separator"
                }`}
              >
                <div className="mb-1 text-muted">
                  #{passage.rank} · {passage.doc_key ?? passage.entry ?? "?"}
                  {passage.version ? ` v${passage.version}` : ""}
                  {passage.relevant["knowledge-hit-v1"] ? " · relevant" : ""}
                </div>
                <p className="line-clamp-4 whitespace-pre-wrap break-words">
                  {passage.text}
                </p>
              </div>
            ))}
        </div>
      ))}
    </div>
  );
}

function Queries({
  data,
  labels,
  ids,
}: {
  data: CompareManyOut;
  labels: string[];
  ids: string[];
}) {
  const [onlyDiffering, setOnlyDiffering] = useState(true);
  const [chosen, setChosen] = useState<string | null>(null);
  const rows = data.queries.filter((row) => !onlyDiffering || row.spread > 0);
  const top = rows.slice(0, 40);
  return (
    <div className="grid gap-3">
      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={onlyDiffering}
          onChange={(event) => setOnlyDiffering(event.target.checked)}
        />
        Only queries the runs disagree on (
        {data.queries.filter((row) => row.spread > 0).length})
      </label>
      {top.length ? (
        <Chart
          option={gridOption(
            top.map((row) => row.item_id),
            top.map((row) => row.values),
            labels,
            data.metrics[0] ?? "",
            { labelCells: false },
          )}
          label={`${data.metrics[0]} per query and run, the widest disagreement first`}
          height={Math.max(200, 18 * top.length + 110)}
          testId="compare-queries"
          renderer="svg"
        />
      ) : null}
      <DataTable
        label="Queries per run"
        rows={rows}
        rowKey={(row) => row.item_id}
        selected={chosen}
        onSelect={(row) => setChosen(row.item_id)}
        columns={[
          { header: "Query", cell: (row) => <code>{row.item_id}</code> },
          ...labels.map((label, index) => ({
            header: label,
            cell: (row: (typeof rows)[number]) =>
              formatValue(row.values[index]),
          })),
          { header: "Spread", cell: (row) => formatValue(row.spread) },
        ]}
      />
      {chosen ? (
        <Section
          title={`The final lists for ${chosen}`}
          description="Relevant passages are outlined."
        >
          <SideBySide ids={ids} labels={labels} itemId={chosen} />
        </Section>
      ) : (
        <p className="text-xs text-muted">
          Select a query to see each run's final list side by side.
        </p>
      )}
    </div>
  );
}

function Latency({
  data,
  labels,
  ids,
}: {
  data: CompareManyOut;
  labels: string[];
  ids: string[];
}) {
  const { api } = useLab();
  const profiles = useLoad(
    () =>
      Promise.all(
        ids.map(async (id) =>
          unwrap(
            await api.GET("/eval-runs/{eval_run_id}/latency", {
              params: { path: { eval_run_id: id } },
            }),
          ),
        ),
      ),
    ids.join(","),
  );
  return (
    <div className="grid gap-6">
      <Chart
        option={latencyOption(data.latency, labels)}
        label="Each run's median milliseconds per step, stacked"
        height={Math.max(200, 50 * labels.length + 70)}
        testId="compare-latency"
        renderer="svg"
      />
      {profiles.data ? (
        <Chart
          option={stepBoxOption(
            profiles.data.map(
              (profile) =>
                profile.steps as unknown as Record<string, StepProfile>,
            ),
            labels,
          )}
          label="Each step's latency as box plots per run: p25 to p75, the median, the minimum to p95"
          height={320}
          testId="compare-boxes"
          renderer="svg"
        />
      ) : profiles.error ? (
        <ErrorNote error={profiles.error} />
      ) : (
        <Loading label="Loading latency profiles" />
      )}
    </div>
  );
}

function Maps({
  labels,
  ids,
  baseline,
}: {
  labels: string[];
  ids: string[];
  baseline: number;
}) {
  const { api } = useLab();
  const maps = useLoad(
    () =>
      Promise.all(
        ids.map(async (id) =>
          unwrap(
            await api.GET("/eval-runs/{eval_run_id}/map", {
              params: { path: { eval_run_id: id } },
            }),
          ),
        ),
      ),
    ids.join(","),
  );
  if (!maps.data)
    return maps.error ? <ErrorNote error={maps.error} /> : <Loading />;
  const base = maps.data[baseline] as MapOut;
  return (
    <div className="grid gap-6">
      {maps.data.map((map, index) => (
        <div key={ids[index]} className="grid gap-2">
          <div className="flex items-center gap-2 text-sm font-semibold">
            <Swatch index={index} />
            {labels[index]}
            {index === baseline ? (
              <span className="text-xs font-normal text-muted">baseline</span>
            ) : (
              <span className="text-xs font-normal text-muted">
                Δ marks a step whose component or settings differ from the
                baseline's
              </span>
            )}
          </div>
          <PipelineMap
            map={map}
            mermaid={map.mermaid}
            changes={index === baseline ? undefined : diffMaps(base, map)}
            label={`The map of ${labels[index]}`}
          />
        </div>
      ))}
    </div>
  );
}

/** The workspace for two to six eval runs. */
export function Compare({ runs }: { runs: string[] }) {
  const { api } = useLab();
  const navigate = useNavigate();
  const [rule, setRule] = useState<Rule>("knowledge-hit-v1");
  const [checkpoint, setCheckpoint] = useState<Checkpoint>("final");
  const [baseline, setBaseline] = useState(runs[0] ?? "");
  const [tab, setTab] = useState<Tab>("Overview");
  const compared = useLoad(
    async () =>
      unwrap(
        await api.GET("/eval-comparisons/many", {
          params: { query: { ids: runs, baseline, rule, checkpoint } },
        }),
      ),
    JSON.stringify([runs, baseline, rule, checkpoint]),
  );
  const data = compared.data;
  const labels = useMemo(() => (data ? runLabels(data.runs) : []), [data]);

  if (runs.length < 2) {
    return (
      <Section title="Compare">
        <Empty>
          Select two to six eval runs in the{" "}
          <Link to={{ name: "evals" }} className="text-link">
            tracking table
          </Link>
          .
        </Empty>
      </Section>
    );
  }
  return (
    <div className="grid gap-4">
      <Section
        title={`Compare ${runs.length} eval runs`}
        description="Paired on the queries every run scored, against the baseline, each run in its own colour."
      >
        <div className="grid gap-3">
          <div className="flex flex-wrap items-end gap-2">
            <label className="grid gap-1 text-xs font-medium">
              Baseline
              <select
                className={inputClass}
                value={baseline}
                onChange={(event) => setBaseline(event.target.value)}
              >
                {runs.map((id, index) => (
                  <option key={id} value={id}>
                    {labels[index] ?? shortHash(id, 8)}
                  </option>
                ))}
              </select>
            </label>
            <label className="grid gap-1 text-xs font-medium">
              Rule
              <select
                className={inputClass}
                value={rule}
                onChange={(event) => setRule(event.target.value as Rule)}
              >
                {RULES.map((name) => (
                  <option key={name}>{name}</option>
                ))}
              </select>
            </label>
            <label className="grid gap-1 text-xs font-medium">
              List
              <select
                className={inputClass}
                value={checkpoint}
                onChange={(event) =>
                  setCheckpoint(event.target.value as Checkpoint)
                }
              >
                {CHECKPOINTS.map((name) => (
                  <option key={name} value={name}>
                    {CHECKPOINT_NAMES[name]}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {data ? (
            <ul className="grid gap-1 text-sm" aria-label="Runs compared">
              {data.runs.map((run, index) => (
                <li
                  key={run.id}
                  className="flex flex-wrap items-center gap-2"
                  data-testid="compared-run"
                >
                  <Swatch index={index} />
                  <Link
                    to={{ name: "eval", evalRunId: run.id }}
                    className="text-link"
                  >
                    {index + 1}. {run.label}
                  </Link>
                  <span className="text-xs text-muted">
                    {String(run.retrieval_config.name)} v
                    {String(run.retrieval_config.version)} ·{" "}
                    {String(run.pipeline_spec.name)} v
                    {String(run.pipeline_spec.version)} ·{" "}
                    {Object.entries(run.models)
                      .map(([role, model]) => `${role} ${model ?? "–"}`)
                      .join(", ")}
                    {formatTags(run.tags) ? ` · ${formatTags(run.tags)}` : ""}
                  </span>
                  {index === data.baseline ? (
                    <span className="rounded bg-default px-1.5 text-xs">
                      baseline
                    </span>
                  ) : null}
                  <button
                    type="button"
                    className="text-xs text-link"
                    aria-label={`Remove ${labels[index]} from the comparison`}
                    onClick={() =>
                      navigate({
                        name: "compare",
                        runs: runs.filter((id) => id !== run.id),
                      })
                    }
                  >
                    remove
                  </button>
                </li>
              ))}
            </ul>
          ) : null}
          {data?.warnings.length ? (
            <p role="status" className="text-sm text-warning">
              {data.warnings
                .map((warning) =>
                  warning === "few_queries"
                    ? "Fewer than 50 queries pair: the intervals are wide."
                    : warning === "different_splits"
                      ? "The runs read different splits: only shared items pair."
                      : warning,
                )
                .join(" ")}
            </p>
          ) : null}
        </div>
      </Section>
      <div
        role="tablist"
        aria-label="Compare views"
        className="flex flex-wrap gap-1"
      >
        {TABS.map((name) => (
          <button
            key={name}
            type="button"
            role="tab"
            aria-selected={tab === name}
            className={`rounded-md px-3 py-1 text-sm ${tab === name ? "bg-default font-semibold" : "hover:bg-default"}`}
            onClick={() => setTab(name)}
          >
            {name}
          </button>
        ))}
      </div>
      <Section title={tab}>
        {data === undefined ? (
          compared.error ? (
            <ErrorNote error={compared.error} />
          ) : (
            <Loading />
          )
        ) : tab === "Overview" ? (
          <Overview data={data} labels={labels} />
        ) : tab === "Charts" ? (
          <Charts data={data} labels={labels} />
        ) : tab === "Parameters" ? (
          <Parameters data={data} labels={labels} />
        ) : tab === "Queries" ? (
          <Queries data={data} labels={labels} ids={runs} />
        ) : tab === "Latency" ? (
          <Latency data={data} labels={labels} ids={runs} />
        ) : (
          <Maps labels={labels} ids={runs} baseline={data.baseline} />
        )}
      </Section>
    </div>
  );
}

export default Compare;
