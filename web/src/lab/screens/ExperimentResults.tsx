import { Button } from "@heroui/react";
import { useEffect, useMemo, useState, type FormEvent } from "react";
import type { components } from "../api";
import { ApiError, unwrap, useLoad } from "../api/load";
import {
  effectsOption,
  heatmapOption,
  paretoOption,
  whereLostOption,
  type EffectRow,
  type Interaction,
  type LostRow,
  type ParetoPoint,
} from "../charts";
import { Chart } from "../components/Chart";
import {
  DataTable,
  Empty,
  ErrorNote,
  Loading,
  PlainText,
  Section,
  StatusChip,
  inputClass,
} from "../components/ui";
import { Link, useLab, useNavigate } from "../context";
import { formatEstimate, type Estimate } from "../evals";
import { routingDocument, routingYaml, type Routing } from "../experiments";
import { formatDuration, isActive, shortHash } from "../format";

type ResultsOut = components["schemas"]["ResultsOut"];
type ConfirmationOut = components["schemas"]["ConfirmationOut"];
type Reason = NonNullable<components["schemas"]["ConfirmIn"]["reason"]>;

const REASONS: { value: Reason; label: string }[] = [
  { value: "run_failed", label: "The first confirmation's run failed" },
  { value: "candidate_changed", label: "Another candidate was chosen on dev" },
  { value: "other", label: "Other (say why in the note)" },
];

interface Guard {
  ordinal: number;
  group: string;
  equivalent: boolean;
  behind: number;
  p_holm: number;
}

function guardsOf(results: ResultsOut): Map<number, Guard> {
  const rows = ((results.guardrails as Row).trials ?? []) as Guard[];
  return new Map(rows.map((row) => [row.ordinal, row]));
}

/** A trial's place against the leader: its group, and ≈ when practically equivalent. */
export function groupLabel(
  guard: Guard | undefined,
  recommended: boolean,
): string {
  if (!guard) return "–";
  const parts = [guard.group === "leader" ? "leader" : guard.group];
  if (guard.equivalent && guard.group !== "leader") parts.push("≈");
  if (recommended) parts.push("recommended");
  return parts.join(" · ");
}
type Row = Record<string, unknown>;

const VIEWS = [
  ["leaderboard", "Leaderboard"],
  ["effects", "Factor effects"],
  ["interactions", "Interactions"],
  ["file-types", "File types"],
  ["pareto", "Pareto"],
  ["where-lost", "Where lost"],
] as const;
type View = (typeof VIEWS)[number][0];

function estimate(value: unknown): Estimate | null {
  return value !== null && typeof value === "object"
    ? (value as Estimate)
    : null;
}

function metricOf(results: ResultsOut): string {
  return String(results.objective.metric);
}

function money(value: unknown): string {
  return typeof value === "number" ? `$${value.toFixed(4)}` : "–";
}

function SampleSize({ results }: { results: ResultsOut }) {
  const guard = results.guardrails as Row;
  const hints = (guard.sample_size ?? null) as Row | null;
  if (!hints) return null;
  const detectable = hints.min_detectable_difference;
  const warnings = (hints.warnings ?? []) as string[];
  return (
    <p className="text-xs text-muted" data-testid="sample-size">
      {String(hints.queries)} queries decide. The leader's interval is{" "}
      {Number(hints.ci_width).toFixed(3)} wide
      {typeof detectable === "number"
        ? `; a paired difference under ${detectable.toFixed(3)} is hard to detect`
        : ""}
      . Groups use Holm's adjustment at α = {String(guard.alpha)}; ≈ marks a
      difference inside ±{String(guard.margin)}, where the cheaper, faster or
      simpler trial is recommended.
      {warnings.includes("few_queries") ? (
        <strong className="text-warning">
          {" "}
          Fewer than 50 queries: decide with care.
        </strong>
      ) : null}
    </p>
  );
}

function Leaderboard({
  results,
  paths,
  onConfirm,
}: {
  results: ResultsOut;
  paths: string[];
  onConfirm: (ordinal: number) => void;
}) {
  const [chosen, setChosen] = useState<string[]>([]);
  const rows = results.leaderboard as Row[];
  const guards = guardsOf(results);
  const recommended = (results.guardrails as Row).recommended;
  const toggle = (evalRunId: string) =>
    setChosen((current) =>
      current.includes(evalRunId)
        ? current.filter((id) => id !== evalRunId)
        : [...current, evalRunId].slice(-2),
    );
  return (
    <div className="grid gap-2" data-testid="view-leaderboard">
      <SampleSize results={results} />
      <DataTable<Row>
        label="Leaderboard"
        rows={rows}
        rowKey={(row) => String(row.ordinal)}
        columns={[
          { header: "Rank", cell: (row) => String(row.rank) },
          { header: "#", cell: (row) => String(row.ordinal) },
          ...paths.map((path) => ({
            header: path,
            cell: (row: Row) =>
              (row.values as Record<string, string>)[path] ?? "–",
          })),
          {
            header: `${metricOf(results)} (95% CI)`,
            cell: (row) => formatEstimate(estimate(row.objective)),
          },
          {
            header: "Against the leader",
            cell: (row) =>
              groupLabel(
                guards.get(Number(row.ordinal)),
                row.ordinal === recommended,
              ),
          },
          {
            header: "Constraints",
            cell: (row) =>
              row.feasible === null || row.feasible === undefined
                ? "–"
                : row.feasible
                  ? "pass"
                  : "fail",
          },
          {
            header: "p95 latency",
            cell: (row) =>
              formatDuration(row.latency_ms_p95 as number | null | undefined),
          },
          {
            header: "Cost / 1k queries",
            cell: (row) => money(row.cost_per_1k_queries_usd),
          },
          {
            header: "Test split",
            cell: (row) => (
              <Button
                size="sm"
                variant="secondary"
                aria-label={`Confirm trial ${String(row.ordinal)} on test`}
                onPress={() => onConfirm(Number(row.ordinal))}
              >
                Confirm on test
              </Button>
            ),
          },
          {
            header: "Compare",
            cell: (row) => {
              const id = String(row.eval_run_id ?? "");
              return (
                <input
                  type="checkbox"
                  aria-label={`Compare trial ${String(row.ordinal)}`}
                  checked={chosen.includes(id)}
                  onChange={() => toggle(id)}
                />
              );
            },
          },
        ]}
      />
      {chosen.length === 2 ? (
        <Link
          to={{
            name: "eval-compare",
            a: chosen[0] ?? null,
            b: chosen[1] ?? null,
          }}
          className="text-link text-sm"
        >
          Compare the two trials query by query
        </Link>
      ) : (
        <p className="text-xs text-muted">
          Choose two trials to compare them query by query.
        </p>
      )}
    </div>
  );
}

function Effects({ results }: { results: ResultsOut }) {
  const effects = results.effects as Row[];
  if (effects.length === 0) return <Empty>No factors.</Empty>;
  return (
    <div className="grid gap-4" data-testid="view-effects">
      {effects.map((factor) => {
        const rows = factor.levels as (EffectRow & Row)[];
        return (
          <div key={String(factor.path)} className="grid gap-2">
            <h3 className="text-sm font-semibold">
              {String(factor.path)}{" "}
              <span className="font-normal text-muted">
                against {String(factor.reference)}
              </span>
            </h3>
            {rows.some((row) => typeof row.delta === "number") ? (
              <Chart
                renderer="svg"
                option={effectsOption(rows, metricOf(results))}
                label={`Effects of ${String(factor.path)} on ${metricOf(results)}`}
                testId="effects-chart"
                height={240}
              />
            ) : null}
            <DataTable<EffectRow & Row>
              label={`Effects of ${String(factor.path)}`}
              rows={rows}
              rowKey={(row) => String(row.level)}
              columns={[
                { header: "Level", cell: (row) => row.label },
                { header: "Pairs", cell: (row) => row.pairs },
                {
                  header: "Change (95% CI)",
                  cell: (row) =>
                    typeof row.delta === "number"
                      ? `${row.delta >= 0 ? "+" : ""}${row.delta.toFixed(3)} (${Number(row.low).toFixed(3)}–${Number(row.high).toFixed(3)})`
                      : "no pair of trials",
                },
                {
                  header: "p",
                  cell: (row) =>
                    typeof row.p_value === "number"
                      ? row.p_value.toFixed(4)
                      : "–",
                },
              ]}
            />
          </div>
        );
      })}
    </div>
  );
}

function Interactions({ results }: { results: ResultsOut }) {
  const pairs = results.interactions as unknown as Interaction[];
  if (pairs.length === 0)
    return <Empty>Interactions need two factors or more.</Empty>;
  return (
    <div className="grid gap-4 md:grid-cols-2" data-testid="view-interactions">
      {pairs.map((pair) => (
        <Chart
          renderer="svg"
          key={`${pair.rows}:${pair.columns}`}
          option={heatmapOption(pair, metricOf(results))}
          label={`Mean ${metricOf(results)} by ${pair.rows} and ${pair.columns}`}
          testId="interaction-heatmap"
          height={300}
        />
      ))}
    </div>
  );
}

function FileTypes({
  results,
  basePipeline,
  experimentName,
}: {
  results: ResultsOut;
  basePipeline: Record<string, unknown> | undefined;
  experimentName: string;
}) {
  const { api, host } = useLab();
  const table = results.file_types as Row;
  const types = (table.types ?? []) as Row[];
  const best = (table.best_trial ?? {}) as Record<string, number>;
  const routing = (table.routing ?? []) as Routing[];
  const [error, setError] = useState<unknown>(null);

  const save = async (route: Routing) => {
    setError(null);
    try {
      if (!basePipeline) throw new Error("no base pipeline");
      const base = unwrap(
        await api.GET("/pipeline-specs/{name}/versions/{version}", {
          params: {
            path: {
              name: String(basePipeline.name),
              version: Number(basePipeline.version),
            },
          },
        }),
      );
      const saved = unwrap(
        await api.POST("/pipeline-specs", {
          body: routingDocument(
            base.document as Record<string, unknown>,
            route,
            experimentName,
          ),
        }),
      );
      host.notify(`Saved ${saved.name} v${saved.version}.`, "success");
    } catch (failure) {
      setError(failure);
    }
  };

  return (
    <div className="grid gap-3" data-testid="view-file-types">
      <DataTable<Row>
        label="File types"
        rows={types}
        rowKey={(row) => String(row.file_type)}
        columns={[
          { header: "Gold file type", cell: (row) => String(row.file_type) },
          { header: "Queries", cell: (row) => String(row.queries) },
          {
            header: "Best trial",
            cell: (row) => {
              const ordinal = best[String(row.file_type)];
              return ordinal === undefined ? "–" : `#${ordinal}`;
            },
          },
          {
            header: `Its ${String(table.metric)} (95% CI)`,
            cell: (row) => {
              const ordinal = best[String(row.file_type)];
              const cells = (row.trials ?? {}) as Record<string, unknown>;
              return ordinal === undefined
                ? "–"
                : formatEstimate(estimate(cells[String(ordinal)]));
            },
          },
        ]}
      />
      {routing.length === 0 ? (
        <p className="text-xs text-muted">
          No factor on the parse stage, so no routing to suggest.
        </p>
      ) : (
        routing.map((route) => (
          <div key={route.path} className="grid gap-2">
            <h3 className="text-sm font-semibold">
              Best {route.path} per file type
            </h3>
            <PlainText
              text={routingYaml(route)}
              label={`Routing from ${route.path}`}
            />
            <div>
              <Button
                size="sm"
                variant="secondary"
                onPress={() => void save(route)}
              >
                Use as routing: save a pipeline
              </Button>
            </div>
          </div>
        ))
      )}
      <ErrorNote error={error} />
    </div>
  );
}

function Pareto({ results }: { results: ResultsOut }) {
  const pareto = results.pareto as unknown as Record<string, ParetoPoint[]>;
  return (
    <div className="grid gap-4 md:grid-cols-2" data-testid="view-pareto">
      <Chart
        renderer="svg"
        option={paretoOption(
          pareto.latency ?? [],
          "p95 latency (ms)",
          metricOf(results),
        )}
        label={`${metricOf(results)} against p95 latency`}
        testId="pareto-chart"
        height={280}
      />
      <Chart
        renderer="svg"
        option={paretoOption(
          pareto.cost ?? [],
          "Cost per 1,000 queries (USD)",
          metricOf(results),
        )}
        label={`${metricOf(results)} against cost per 1,000 queries`}
        testId="pareto-chart"
        height={280}
      />
    </div>
  );
}

function WhereLost({ results }: { results: ResultsOut }) {
  const lost = results.where_lost as Row;
  const rows = (lost.trials ?? []) as unknown as LostRow[];
  const hardest = (lost.hardest ?? []) as Row[];
  const top = (results.leaderboard as Row[])[0];
  return (
    <div className="grid gap-3" data-testid="view-where-lost">
      <Chart
        renderer="svg"
        option={whereLostOption(rows)}
        label="Queries found and lost by stage, per trial"
        testId="where-lost-chart"
        height={280}
      />
      <DataTable<Row>
        label="Queries most trials missed"
        rows={hardest}
        rowKey={(row) => String(row.item_id)}
        columns={[
          {
            header: "Query",
            cell: (row) =>
              top?.eval_run_id ? (
                <Link
                  to={{
                    name: "eval-query",
                    evalRunId: String(top.eval_run_id),
                    itemId: String(row.item_id),
                  }}
                  className="text-link"
                >
                  {String(row.item_id)}
                </Link>
              ) : (
                String(row.item_id)
              ),
          },
          {
            header: "Trials that missed it",
            cell: (row) => String(row.missed_by),
          },
          {
            header: "Lost at",
            cell: (row) =>
              Object.entries((row.lost_at ?? {}) as Record<string, number>)
                .map(([stage, count]) => `${stage} ${count}`)
                .join(", "),
          },
        ]}
      />
    </div>
  );
}

/** The experiment's reads of the test split: each trial's dev and test results. */
function Confirmations({
  experimentId,
  experimentName,
  reloadKey,
  metric,
}: {
  experimentId: string;
  experimentName: string;
  reloadKey: number;
  metric: string;
}) {
  const { api } = useLab();
  const navigate = useNavigate();
  const [tick, setTick] = useState(0);
  const [exporting, setExporting] = useState<ConfirmationOut | null>(null);
  const [name, setName] = useState(`kp-${experimentName}`.slice(0, 64));
  const [description, setDescription] = useState("");
  const [error, setError] = useState<unknown>(null);

  const exportProfile = async (event: FormEvent) => {
    event.preventDefault();
    if (exporting === null) return;
    setError(null);
    try {
      const profile = unwrap(
        await api.POST("/profiles", {
          body: {
            experiment_id: experimentId,
            trial_id: exporting.trial_id,
            name: name.trim(),
            description,
          },
        }),
      );
      navigate({ name: "profile", profileId: profile.id });
    } catch (failure) {
      setError(failure);
    }
  };
  const found = useLoad(
    async () =>
      unwrap(
        await api.GET("/experiments/{experiment_id}/confirmations", {
          params: { path: { experiment_id: experimentId } },
        }),
      ),
    `${experimentId}:${reloadKey}:${tick}`,
  );
  const items = found.data?.items ?? [];
  const active = items.some((item) => isActive(item.status));
  useEffect(() => {
    if (!active) return;
    const timer = setTimeout(() => setTick((value) => value + 1), 1000);
    return () => clearTimeout(timer);
  }, [active, found.data]);
  if (found.data === undefined) {
    return found.error ? <ErrorNote error={found.error} /> : null;
  }
  return (
    <div className="grid gap-2" data-testid="confirmations">
      <h3 className="text-sm font-semibold">
        Test-split reads{" "}
        <span className="font-normal text-muted" data-testid="test-reads">
          (the eval set's test split has been read {found.data.test_reads}{" "}
          {found.data.test_reads === 1 ? "time" : "times"})
        </span>
      </h3>
      {items.length === 0 ? (
        <Empty>
          No trial has been confirmed on the test split. Choose on dev, then
          confirm once.
        </Empty>
      ) : (
        <DataTable<ConfirmationOut>
          label="Test-split reads"
          rows={items}
          rowKey={(item) => item.id}
          columns={[
            { header: "Read", cell: (item) => item.ordinal },
            { header: "Trial", cell: (item) => `#${item.trial_ordinal}` },
            {
              header: `${metric} on dev`,
              cell: (item) => formatEstimate(estimate(item.dev)),
            },
            {
              header: `${metric} on test`,
              cell: (item) => formatEstimate(estimate(item.test)),
            },
            {
              header: "Status",
              cell: (item) => <StatusChip status={item.status} />,
            },
            {
              header: "Reason",
              cell: (item) => item.reason?.replace(/_/g, " ") ?? "first read",
            },
            {
              header: "Profile",
              cell: (item) =>
                item.status === "succeeded" ? (
                  <Button
                    size="sm"
                    variant="secondary"
                    aria-label={`Export read ${item.ordinal} as a profile`}
                    onPress={() => setExporting(item)}
                  >
                    Export as profile
                  </Button>
                ) : (
                  "–"
                ),
            },
          ]}
        />
      )}
      {exporting === null ? null : (
        <form
          onSubmit={exportProfile}
          aria-label="Export a knowledge profile"
          className="grid gap-2 rounded-md border border-separator p-3"
        >
          <p className="text-sm">
            Export trial #{exporting.trial_ordinal}, confirmed on test, as a
            knowledge profile version.
          </p>
          <label className="grid gap-1 text-xs font-medium">
            Profile name
            <input
              className={inputClass}
              value={name}
              onChange={(event) => setName(event.target.value)}
              pattern="[a-z0-9][a-z0-9._\-]{0,63}"
              required
            />
          </label>
          <label className="grid gap-1 text-xs font-medium">
            Description
            <input
              className={inputClass}
              value={description}
              maxLength={2000}
              onChange={(event) => setDescription(event.target.value)}
            />
          </label>
          <div>
            <Button type="submit" size="sm">
              Export the profile
            </Button>
          </div>
          <ErrorNote error={error} />
        </form>
      )}
    </div>
  );
}

/** The results views. */
export function ExperimentResults({
  experimentId,
  experimentName,
  progressKey,
  paths,
  basePipeline,
  trialIds,
}: {
  experimentId: string;
  experimentName: string;
  progressKey: string;
  paths: string[];
  basePipeline: Record<string, unknown> | undefined;
  trialIds: Record<number, string>;
}) {
  const { api, host } = useLab();
  const [view, setView] = useState<View>("leaderboard");
  const [reads, setReads] = useState(0);
  const [rereading, setRereading] = useState<number | null>(null);
  const [reason, setReason] = useState<Reason>("candidate_changed");
  const [note, setNote] = useState("");
  const [confirmError, setConfirmError] = useState<unknown>(null);

  const confirmTrial = async (
    ordinal: number,
    logged?: { reason: Reason; note: string },
  ) => {
    const trialId = trialIds[ordinal];
    if (trialId === undefined) return;
    if (logged === undefined) {
      const accepted = await host.confirm({
        title: "Read the locked test split?",
        body: `Trial ${ordinal} is evaluated once on the test split. The read is counted and logged; another read for this experiment needs a reason.`,
        confirmLabel: "Read the test split",
      });
      if (!accepted) return;
    }
    setConfirmError(null);
    try {
      unwrap(
        await api.POST(
          "/experiments/{experiment_id}/trials/{trial_id}/confirm",
          {
            params: {
              path: { experiment_id: experimentId, trial_id: trialId },
            },
            body: {
              confirm_test_read: true,
              ...(logged
                ? { reason: logged.reason, note: logged.note.trim() || null }
                : {}),
            },
          },
        ),
      );
      setRereading(null);
      setNote("");
      setReads((value) => value + 1);
      host.notify(
        `Trial ${ordinal} is being evaluated on the test split.`,
        "info",
      );
    } catch (failure) {
      if (
        failure instanceof ApiError &&
        failure.code === "test_split_already_read"
      ) {
        setRereading(ordinal);
      } else {
        setConfirmError(failure);
      }
    }
  };

  const reread = (event: FormEvent) => {
    event.preventDefault();
    if (rereading !== null) void confirmTrial(rereading, { reason, note });
  };
  const results = useLoad(
    async () =>
      unwrap(
        await api.GET("/experiments/{experiment_id}/results", {
          params: { path: { experiment_id: experimentId } },
        }),
      ),
    `${experimentId}:${progressKey}`,
  );
  const data = results.data;
  const content = useMemo(() => {
    if (data === undefined) return null;
    switch (view) {
      case "leaderboard":
        return (
          <Leaderboard
            results={data}
            paths={paths}
            onConfirm={(ordinal) => void confirmTrial(ordinal)}
          />
        );
      case "effects":
        return <Effects results={data} />;
      case "interactions":
        return <Interactions results={data} />;
      case "file-types":
        return (
          <FileTypes
            results={data}
            basePipeline={basePipeline}
            experimentName={experimentName}
          />
        );
      case "pareto":
        return <Pareto results={data} />;
      case "where-lost":
        return <WhereLost results={data} />;
    }
    // confirmTrial changes with each render; the views don't depend on it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, view, paths, basePipeline, experimentName, trialIds]);

  return (
    <Section
      title="Results"
      description={
        data
          ? `${data.trials} trials, ${data.queries} queries with evidence; ${metricOf(data)} (${String(data.objective.rule)}, ${String(data.objective.checkpoint)}), ${String(data.objective.goal)}.`
          : "From the trials that have finished."
      }
      actions={
        <div className="flex flex-wrap gap-1" role="group" aria-label="Views">
          {VIEWS.map(([name, label]) => (
            <Button
              key={name}
              size="sm"
              variant={name === view ? "primary" : "secondary"}
              aria-pressed={name === view}
              onPress={() => setView(name)}
            >
              {label}
            </Button>
          ))}
        </div>
      }
    >
      {data === undefined ? (
        results.error ? (
          <ErrorNote error={results.error} />
        ) : (
          <Loading />
        )
      ) : data.trials === 0 ? (
        <Empty>No trial has finished yet.</Empty>
      ) : (
        content
      )}
      {rereading === null ? null : (
        <form
          onSubmit={reread}
          aria-label="Read the test split again"
          className="mt-3 grid gap-2 rounded-md border border-warning p-3"
        >
          <p role="alert" className="text-sm">
            The test split was read for this experiment already. Reading it
            again for trial {rereading} needs a reason, which is logged.
          </p>
          <label className="grid gap-1 text-xs font-medium">
            Reason
            <select
              className={inputClass}
              value={reason}
              onChange={(event) => setReason(event.target.value as Reason)}
            >
              {REASONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label className="grid gap-1 text-xs font-medium">
            Note (kept with the read, never logged)
            <textarea
              className={inputClass}
              maxLength={500}
              value={note}
              onChange={(event) => setNote(event.target.value)}
            />
          </label>
          <div className="flex gap-2">
            <Button type="submit" size="sm" variant="danger">
              Read again (logged)
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onPress={() => setRereading(null)}
            >
              Don't read it
            </Button>
          </div>
        </form>
      )}
      <ErrorNote error={confirmError} />
      {data && data.trials > 0 ? (
        <div className="mt-4">
          <Confirmations
            experimentId={experimentId}
            experimentName={experimentName}
            reloadKey={reads}
            metric={metricOf(data)}
          />
        </div>
      ) : null}
      <p className="mt-2 text-xs text-muted">
        Eval runs:{" "}
        {(data?.leaderboard as Row[] | undefined)
          ?.slice(0, 3)
          .map((row) => shortHash(String(row.eval_run_id), 8))
          .join(", ") ?? "–"}
        {data && data.leaderboard.length > 3 ? "…" : ""}
      </p>
    </Section>
  );
}
