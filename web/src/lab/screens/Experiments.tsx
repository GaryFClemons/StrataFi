import { Button } from "@heroui/react";
import { Suspense, lazy, useMemo, useState, type FormEvent } from "react";
import type { components } from "../api";
import { unwrap, useLoad } from "../api/load";
import { useLiveExperiment } from "../components/live";
import {
  DataTable,
  Empty,
  ErrorNote,
  Loading,
  Section,
  StatusChip,
  inputClass,
} from "../components/ui";
import { Link, useLab, useNavigate } from "../context";
import { formatEstimate } from "../evals";
import {
  EXAMPLE_EXPERIMENT,
  budgetRows,
  factorPaths,
  halvingRungs,
  isExperimentActive,
  rungTrail,
  sharedShare,
  workRows,
  type HalvingRung,
  type StageWork,
} from "../experiments";
import { formatDuration, formatTime, shortHash } from "../format";

// The results views draw charts (ECharts), so they load only when an
// experiment has results to show, in a chunk of their own.
const ExperimentPanels = lazy(() =>
  import("./Panels").then((module) => ({ default: module.ExperimentPanels })),
);
const ExperimentResults = lazy(() =>
  import("./ExperimentResults").then((module) => ({
    default: module.ExperimentResults,
  })),
);

type ExperimentOut = components["schemas"]["ExperimentOut"];
type PlanOut = components["schemas"]["PlanOut"];
type TrialOut = components["schemas"]["TrialOut"];
type Plan = Record<string, unknown>;

function counts(experiment: ExperimentOut): string {
  const trials = (experiment.counts.trials ?? {}) as Record<string, number>;
  const parts = Object.entries(trials).map(
    ([status, count]) => `${count} ${status}`,
  );
  return parts.length > 0 ? parts.join(", ") : "–";
}

function spent(experiment: ExperimentOut): string {
  const total = (experiment.spent as Record<string, unknown>).total_usd;
  return typeof total === "number" ? `$${total.toFixed(4)}` : "–";
}

/** The plan to review: trials, unique work per stage, estimate and budget. */
export function PlanSummary({ plan }: { plan: Plan }) {
  const rows = workRows(plan);
  const estimate = (plan.estimate ?? {}) as Record<string, unknown>;
  const cost = (estimate.cost_usd ?? {}) as Record<string, number>;
  const evaluation = (estimate.eval ?? {}) as Record<string, number>;
  const work = (plan.work ?? {}) as Record<string, unknown>;
  const warnings = (plan.warnings ?? []) as string[];
  const rungs = halvingRungs(plan);
  const search = (plan.search ?? {}) as Record<string, unknown>;
  return (
    <div className="grid gap-3" data-testid="plan-summary">
      <p className="text-sm">
        <strong>{String(plan.trials)}</strong> trials ({String(plan.strategy)};{" "}
        {String(plan.grid)} combinations, {String(plan.excluded)} excluded) over{" "}
        <strong>{String(work.pipelines)}</strong> pipeline runs. Sharing saves{" "}
        {(sharedShare(rows) * 100).toFixed(0)}% of the stage work.
      </p>
      <DataTable<StageWork>
        label="Unique work"
        rows={rows}
        rowKey={(row) => row.stage}
        columns={[
          { header: "Stage", cell: (row) => row.stage },
          { header: "Every trial alone", cell: (row) => row.tasks },
          { header: "Unique", cell: (row) => row.unique },
          { header: "Cached", cell: (row) => row.cached },
          { header: "To run", cell: (row) => row.to_run },
        ]}
      />
      <p className="text-sm" data-testid="plan-estimate">
        Estimate: {formatDuration(Number(estimate.wall_seconds ?? 0) * 1000)} of
        work, ${Number(cost.total ?? 0).toFixed(4)} (ingest $
        {Number(cost.ingest ?? 0).toFixed(4)}, evaluation $
        {Number(cost.eval ?? 0).toFixed(4)}); {evaluation.to_run ?? 0} of{" "}
        {evaluation.trials ?? 0} trials to evaluate on {evaluation.queries ?? 0}{" "}
        queries.
      </p>
      {rungs.length > 0 ? (
        <div className="grid gap-2" data-testid="plan-halving">
          <p className="text-sm">
            Successive halving scores every trial on a seeded sample of the
            queries first, and each rung keeps the best 1/
            {String(search.eta)} for a bigger one; a trial is pruned only when
            it's clearly behind the rung's leader, so a rung can keep more.
            Nominally it scores {(Number(search.share ?? 0) * 100).toFixed(0)}%
            of the grid's queries.
          </p>
          <DataTable<HalvingRung>
            label="Rungs"
            rows={rungs}
            rowKey={(row) => String(row.rung)}
            columns={[
              { header: "Rung", cell: (row) => row.rung },
              { header: "Queries", cell: (row) => row.queries },
              {
                header: "Keeps at least",
                cell: (row) =>
                  row.rung === rungs.length ? "every trial" : row.keep,
              },
            ]}
          />
        </div>
      ) : null}
      <DataTable
        label="Budget"
        rows={budgetRows(plan)}
        rowKey={(row) => row.name}
        columns={[
          { header: "Budget", cell: (row) => row.label },
          { header: "Planned", cell: (row) => row.planned },
          { header: "Limit", cell: (row) => row.max },
          {
            header: "",
            cell: (row) =>
              row.over ? <StatusChip status="failed" label="over" /> : null,
          },
        ]}
      />
      {warnings.length > 0 ? (
        <p className="text-sm text-warning">Warnings: {warnings.join(", ")}</p>
      ) : null}
    </div>
  );
}

/** Experiments, and a new one from YAML: create it, then review its plan. */
export function Experiments() {
  const { api } = useLab();
  const navigate = useNavigate();
  const experiments = useLoad(
    async () =>
      unwrap(
        await api.GET("/experiments", { params: { query: { limit: 50 } } }),
      ),
    "experiments",
  );
  const [content, setContent] = useState(EXAMPLE_EXPERIMENT);
  const [planned, setPlanned] = useState<PlanOut | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const check = async () => {
    setBusy(true);
    setError(null);
    try {
      setPlanned(
        unwrap(await api.POST("/experiments/plan", { body: { content } })),
      );
    } catch (failure) {
      setError(failure);
    } finally {
      setBusy(false);
    }
  };

  const save = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const created = unwrap(
        await api.POST("/experiments", { body: { content } }),
      );
      navigate({ name: "experiment", experimentId: created.id });
    } catch (failure) {
      setError(failure);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid gap-4">
      <Section
        title="New experiment"
        description="Factors are paths into a base pipeline spec or retrieval config, each with its levels. Check the plan, save it, then start it from its page."
      >
        <form
          onSubmit={save}
          className="grid gap-2"
          aria-label="New experiment"
        >
          <label className="grid gap-1 text-xs font-medium">
            Experiment (YAML, stratafi.experiment.v1)
            <textarea
              className={`${inputClass} min-h-72 font-mono text-xs`}
              value={content}
              onChange={(event) => {
                setContent(event.target.value);
                setPlanned(null);
              }}
              spellCheck={false}
            />
          </label>
          <div className="flex flex-wrap gap-2">
            <Button
              size="sm"
              variant="secondary"
              onPress={check}
              isDisabled={busy}
            >
              Check plan
            </Button>
            <Button
              type="submit"
              size="sm"
              isDisabled={busy || (planned !== null && !planned.valid)}
            >
              Save for review
            </Button>
          </div>
        </form>
        <ErrorNote error={error} />
        {planned === null ? null : planned.valid && planned.plan ? (
          <div className="mt-3">
            <PlanSummary plan={planned.plan} />
          </div>
        ) : (
          <ul className="mt-3 text-sm text-danger" aria-label="Problems">
            {planned.issues.map((issue) => (
              <li key={`${issue.path}:${issue.code}`}>
                <code>{issue.path}</code>: {issue.code}
              </li>
            ))}
          </ul>
        )}
      </Section>
      <Section title="Experiments" description="Newest first.">
        {experiments.data === undefined ? (
          experiments.error ? (
            <ErrorNote error={experiments.error} />
          ) : (
            <Loading />
          )
        ) : experiments.data.items.length === 0 ? (
          <Empty>No experiments yet.</Empty>
        ) : (
          <DataTable<ExperimentOut>
            label="Experiments"
            rows={experiments.data.items}
            rowKey={(experiment) => experiment.id}
            columns={[
              {
                header: "Experiment",
                cell: (experiment) => (
                  <Link
                    to={{ name: "experiment", experimentId: experiment.id }}
                    className="text-link"
                  >
                    {experiment.name}
                  </Link>
                ),
              },
              {
                header: "Status",
                cell: (experiment) => <StatusChip status={experiment.status} />,
              },
              { header: "Trials", cell: counts },
              { header: "Spent", cell: spent },
              {
                header: "Created",
                cell: (experiment) => formatTime(experiment.created_at),
              },
            ]}
          />
        )}
      </Section>
    </div>
  );
}

function TrialsTable({
  trials,
  paths,
  objective,
}: {
  trials: TrialOut[];
  paths: string[];
  objective: string;
}) {
  return (
    <DataTable<TrialOut>
      label="Trials"
      rows={trials}
      rowKey={(trial) => trial.id}
      columns={[
        { header: "#", cell: (trial) => trial.ordinal },
        ...paths.map((path) => ({
          header: path,
          cell: (trial: TrialOut) => trial.values[path] ?? "–",
        })),
        {
          header: "Status",
          cell: (trial) => (
            <StatusChip
              status={trial.status}
              label={
                trial.error_code
                  ? `${trial.status} (${trial.error_code.replace(/_/g, " ")})`
                  : undefined
              }
            />
          ),
        },
        {
          header: `${objective} (95% CI)`,
          cell: (trial) =>
            formatEstimate(
              trial.objective as Parameters<typeof formatEstimate>[0],
            ),
        },
        ...(trials.some((trial) => trial.rungs.length > 0)
          ? [
              {
                header: "Queries by rung",
                cell: (trial: TrialOut) => rungTrail(trial.rungs),
              },
            ]
          : []),
        {
          header: "Constraints",
          cell: (trial) =>
            trial.feasible === null ? "–" : trial.feasible ? "pass" : "fail",
        },
        {
          header: "Eval run",
          cell: (trial) =>
            trial.eval_run_id ? (
              <Link
                to={{ name: "eval", evalRunId: trial.eval_run_id }}
                className="text-link"
              >
                <code>{shortHash(trial.eval_run_id, 8)}</code>
              </Link>
            ) : (
              "–"
            ),
        },
      ]}
    />
  );
}

/** One experiment: its plan to review and start, then its trials as they finish. */
export function ExperimentDetail({ experimentId }: { experimentId: string }) {
  const { api, host } = useLab();
  const live = useLiveExperiment(experimentId);
  const experiment = live.data;
  const progressKey = experiment
    ? `${experiment.status}:${JSON.stringify(experiment.counts)}`
    : "";
  const trials = useLoad(
    async () =>
      unwrap(
        await api.GET("/experiments/{experiment_id}/trials", {
          params: { path: { experiment_id: experimentId } },
        }),
      ),
    `${experimentId}:${progressKey}`,
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const trialIds = useMemo(
    () =>
      Object.fromEntries(
        (trials.data?.items ?? []).map((trial) => [trial.ordinal, trial.id]),
      ),
    [trials.data],
  );

  if (experiment === undefined) {
    return live.error ? <ErrorNote error={live.error} /> : <Loading />;
  }
  const plan = experiment.plan as Plan;
  const document = experiment.document as Record<string, unknown>;
  const objective = String(
    (document.objective as Record<string, unknown> | undefined)?.metric ??
      "objective",
  );

  const start = async () => {
    const confirmed = await host.confirm({
      title: "Start this experiment?",
      body: `${String(plan.trials)} trials over ${String(
        (plan.work as Record<string, unknown> | undefined)?.pipelines,
      )} pipeline runs. Its runs stop at the budget.`,
      confirmLabel: "Start",
    });
    if (!confirmed) return;
    setBusy(true);
    setError(null);
    try {
      unwrap(
        await api.POST("/experiments/{experiment_id}/start", {
          params: { path: { experiment_id: experimentId } },
          body: { confirm: true },
        }),
      );
      live.reload();
    } catch (failure) {
      setError(failure);
    } finally {
      setBusy(false);
    }
  };

  const cancel = async () => {
    const confirmed = await host.confirm({
      title: "Cancel this experiment?",
      body: "Nothing new starts, and its runs in flight are cancelled. Finished trials stay.",
      confirmLabel: "Cancel experiment",
      danger: true,
    });
    if (!confirmed) return;
    setBusy(true);
    setError(null);
    try {
      unwrap(
        await api.POST("/experiments/{experiment_id}/cancel", {
          params: { path: { experiment_id: experimentId } },
        }),
      );
      live.reload();
    } catch (failure) {
      setError(failure);
    } finally {
      setBusy(false);
    }
  };

  const cancellable =
    experiment.status === "planned" || isExperimentActive(experiment.status);
  return (
    <div className="grid gap-4">
      <p className="text-sm">
        <Link to={{ name: "experiments" }} className="text-link">
          Experiments
        </Link>{" "}
        / {experiment.name}
      </p>
      <Section
        title={experiment.name}
        description={`${String(document.corpus ?? "")} · objective ${objective}`}
        actions={
          <>
            <StatusChip status={experiment.status} />
            {experiment.status === "planned" ? (
              <Button size="sm" onPress={start} isDisabled={busy}>
                Start
              </Button>
            ) : null}
            {cancellable ? (
              <Button
                size="sm"
                variant="danger"
                onPress={cancel}
                isDisabled={busy}
              >
                Cancel
              </Button>
            ) : null}
          </>
        }
      >
        <div
          className="flex flex-wrap gap-6 text-sm"
          data-testid="experiment-progress"
        >
          <span>Trials: {counts(experiment)}</span>
          <span>Spent: {spent(experiment)}</span>
          {experiment.stop_reason ? (
            <span>Stopped by: {experiment.stop_reason}</span>
          ) : null}
          <span>Started: {formatTime(experiment.started_at)}</span>
        </div>
        <ErrorNote error={error} />
      </Section>
      <Section
        title="Plan"
        description="Shared prefixes run once: later runs find them cached."
      >
        <PlanSummary plan={plan} />
      </Section>
      {experiment.status === "planned" ? null : (
        <Suspense fallback={<Loading label="Loading the results views" />}>
          <ExperimentResults
            experimentId={experimentId}
            experimentName={experiment.name}
            progressKey={progressKey}
            paths={factorPaths(plan)}
            basePipeline={
              (plan.base as Record<string, Record<string, unknown>> | undefined)
                ?.pipeline
            }
            trialIds={trialIds}
          />
        </Suspense>
      )}
      <Section
        title="Trials"
        description="In plan order; each fills in as it finishes."
      >
        {trials.data === undefined ? (
          trials.error ? (
            <ErrorNote error={trials.error} />
          ) : (
            <Loading />
          )
        ) : (
          <TrialsTable
            trials={trials.data.items}
            paths={factorPaths(plan)}
            objective={objective}
          />
        )}
      </Section>
      <Suspense fallback={<Loading label="Loading the experiment's note" />}>
        <ExperimentPanels experimentId={experimentId} />
      </Suspense>
    </div>
  );
}
