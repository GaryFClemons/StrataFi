import { Button } from "@heroui/react";
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";
import type { components } from "../api";
import { ApiError, unwrap, useLoad } from "../api/load";
import { Chart } from "../components/Chart";
import { useLiveTestRun } from "../components/live";
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
import { formatTime } from "../format";
import { downloadText } from "../profiles";
import {
  EXAMPLE_SUITE,
  SCOPES,
  applyCompletion,
  checkIds,
  checkTrend,
  completionsAt,
  evidenceRoute,
  passShares,
  thresholdText,
  type EvidenceLink,
  type HistoryRun,
} from "../suites";

type SuiteOut = components["schemas"]["SuiteOut"];
type TestRunOut = components["schemas"]["TestRunOut"];
type CheckOut = components["schemas"]["CheckOut"];
type ValidationOut = components["schemas"]["ValidationOut"];
type TokenOut = components["schemas"]["TokenOut"];
type WebhookOut = components["schemas"]["WebhookOut"];
type ScheduleOut = components["schemas"]["ScheduleOut"];
type CodeFileOut = components["schemas"]["CodeFileOut"];

const VALIDATE_AFTER_MILLISECONDS = 400;

function runSummary(run: TestRunOut): string {
  const summary = run.summary as Record<string, number>;
  if (!summary.checks) return "–";
  return `${summary.passed ?? 0} passed, ${summary.failed ?? 0} failed, ${
    summary.errored ?? 0
  } errors, ${summary.skipped ?? 0} skipped`;
}

/** A suite's last runs as a line of their passing share: high is good. */
export function Sparkline({
  values,
  label,
}: {
  values: number[];
  label: string;
}) {
  if (values.length === 0) return <span className="text-muted">–</span>;
  const width = 80;
  const height = 20;
  const step = values.length > 1 ? width / (values.length - 1) : 0;
  const points = values
    .map((value, index) => `${index * step},${height - value * height}`)
    .join(" ");
  return (
    <svg
      width={width}
      height={height}
      role="img"
      aria-label={label}
      data-testid="suite-trend"
      className="overflow-visible"
    >
      {values.length > 1 ? (
        <polyline
          points={points}
          fill="none"
          stroke="currentColor"
          strokeWidth={1.5}
        />
      ) : null}
      {values.map((value, index) => (
        <circle
          key={index}
          cx={index * step}
          cy={height - value * height}
          r={2}
          fill={value === 1 ? "var(--success, green)" : "var(--danger, red)"}
        />
      ))}
    </svg>
  );
}

function SuiteTrend({ name }: { name: string }) {
  const { api } = useLab();
  const history = useLoad(
    async () =>
      unwrap(
        await api.GET("/test-suites/{name}/history", {
          params: { path: { name }, query: { limit: 20 } },
        }),
      ),
    `history:${name}`,
  );
  const runs = (history.data?.runs ?? []) as unknown as HistoryRun[];
  const last = runs[runs.length - 1];
  return (
    <div className="flex items-center gap-2">
      {last ? <StatusChip status={last.status} /> : <span>never run</span>}
      <Sparkline
        values={passShares(runs)}
        label={`${name}: passing share of its last ${runs.length} runs`}
      />
    </div>
  );
}

/**
 * A suite in YAML, checked by the server as it's typed, with completions for
 * what the cursor's place in stratafi.test-suite.v1 takes. Tab writes the
 * first completion.
 */
export function SuiteEditor({
  initial,
  label = "Suite (YAML, stratafi.test-suite.v1)",
}: {
  initial: string;
  label?: string;
}) {
  const { api } = useLab();
  const navigate = useNavigate();
  const [content, setContent] = useState(initial);
  const [cursor, setCursor] = useState(0);
  const [verdict, setVerdict] = useState<ValidationOut | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [saved, setSaved] = useState<SuiteOut | null>(null);
  const editor = useRef<HTMLTextAreaElement>(null);
  const completion = useMemo(
    () => completionsAt(content, cursor),
    [content, cursor],
  );

  useEffect(() => {
    let current = true;
    const timer = setTimeout(() => {
      void api
        .POST("/test-suites/validate", { body: { content } })
        .then((result) => {
          if (current && result.data) setVerdict(result.data);
        });
    }, VALIDATE_AFTER_MILLISECONDS);
    return () => {
      current = false;
      clearTimeout(timer);
    };
  }, [api, content]);

  const complete = (option: string) => {
    const done = applyCompletion(content, completion, option);
    setContent(done.text);
    setCursor(done.cursor);
    requestAnimationFrame(() => {
      editor.current?.focus();
      editor.current?.setSelectionRange(done.cursor, done.cursor);
    });
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    const first = completion.options[0];
    if (event.key === "Tab" && !event.shiftKey && first !== undefined) {
      event.preventDefault();
      complete(first);
    }
  };

  const save = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      setSaved(unwrap(await api.POST("/test-suites", { body: { content } })));
    } catch (failure) {
      setError(failure);
    } finally {
      setBusy(false);
    }
  };

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const started = unwrap(
        await api.POST("/test-runs", { body: { content, trigger: "api" } }),
      );
      navigate({ name: "test-run", testRunId: started.id });
    } catch (failure) {
      setError(failure);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid gap-2">
      <form onSubmit={save} className="grid gap-2" aria-label="Suite editor">
        <label className="grid gap-1 text-xs font-medium">
          {label}
          <textarea
            ref={editor}
            className={`${inputClass} min-h-72 font-mono text-xs`}
            value={content}
            onChange={(event) => {
              setContent(event.target.value);
              setCursor(event.target.selectionStart);
              setSaved(null);
            }}
            onSelect={(event) => setCursor(event.currentTarget.selectionStart)}
            onKeyDown={onKeyDown}
            spellCheck={false}
          />
        </label>
        {completion.options.length > 0 ? (
          <div
            className="flex flex-wrap items-center gap-1 text-xs"
            aria-label="Completions"
            role="group"
          >
            <span className="text-muted">Tab or pick:</span>
            {completion.options.slice(0, 12).map((option) => (
              <button
                key={option}
                type="button"
                className="rounded border border-separator px-1.5 py-0.5 font-mono hover:bg-default"
                onClick={() => complete(option)}
              >
                {option}
              </button>
            ))}
          </div>
        ) : null}
        <div className="flex flex-wrap gap-2">
          <Button type="submit" size="sm" variant="secondary" isDisabled={busy}>
            Save
          </Button>
          <Button
            size="sm"
            onPress={() => void run()}
            isDisabled={busy || verdict?.valid === false}
          >
            Save and run
          </Button>
        </div>
      </form>
      <ErrorNote error={error} />
      {saved ? (
        <p className="text-sm" role="status">
          Saved{" "}
          <Link
            to={{ name: "suite", suiteName: saved.name }}
            className="text-link"
          >
            {saved.name} v{saved.version}
          </Link>
          .
        </p>
      ) : null}
      {verdict === null ? null : verdict.valid ? (
        <p className="text-sm text-success" data-testid="suite-valid">
          Valid.
          {verdict.needs.eval_run ? " Its checks score queries." : ""}
          {verdict.needs.baseline ? " Some compare with a baseline." : ""}
        </p>
      ) : (
        <ul className="text-sm text-danger" aria-label="Problems">
          {verdict.issues.map((issue) => (
            <li key={`${issue.path}:${issue.code}`}>
              <code>{issue.path}</code>: {issue.code.replace(/_/g, " ")}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function RunsTable({ runs, label }: { runs: TestRunOut[]; label: string }) {
  return (
    <DataTable<TestRunOut>
      label={label}
      rows={runs}
      rowKey={(run) => run.id}
      columns={[
        {
          header: "Test run",
          cell: (run) => (
            <Link
              to={{ name: "test-run", testRunId: run.id }}
              className="text-link font-mono"
            >
              {run.id.slice(0, 8)}
            </Link>
          ),
        },
        {
          header: "Suite",
          cell: (run) => `${run.suite_name} v${run.suite_version}`,
        },
        { header: "Status", cell: (run) => <StatusChip status={run.status} /> },
        { header: "Trigger", cell: (run) => run.triggers.join(", ") },
        { header: "Checks", cell: runSummary },
        { header: "Created", cell: (run) => formatTime(run.created_at) },
      ]}
    />
  );
}

/** Suites with their last status and trend, a new suite, and recent runs. */
export function Tests() {
  const { api } = useLab();
  const suites = useLoad(
    async () =>
      unwrap(
        await api.GET("/test-suites", { params: { query: { limit: 100 } } }),
      ),
    "suites",
  );
  const runs = useLoad(
    async () =>
      unwrap(await api.GET("/test-runs", { params: { query: { limit: 20 } } })),
    "test-runs",
  );
  return (
    <div className="grid gap-4">
      <Section
        title="Test suites"
        description="Each suite's latest version, its last result and how its last runs went."
      >
        <p className="mb-2 text-sm">
          <Link to={{ name: "automation" }} className="text-link">
            Automation: schedules, triggers, API tokens and webhooks
          </Link>
        </p>
        {suites.data === undefined ? (
          suites.error ? (
            <ErrorNote error={suites.error} />
          ) : (
            <Loading />
          )
        ) : suites.data.items.length === 0 ? (
          <Empty>No suites yet. Write one below.</Empty>
        ) : (
          <DataTable<SuiteOut>
            label="Suites"
            rows={suites.data.items}
            rowKey={(suite) => suite.id}
            columns={[
              {
                header: "Suite",
                cell: (suite) => (
                  <Link
                    to={{ name: "suite", suiteName: suite.name }}
                    className="text-link"
                  >
                    {suite.name}
                  </Link>
                ),
              },
              { header: "Version", cell: (suite) => `v${suite.version}` },
              { header: "Checks", cell: (suite) => suite.checks },
              {
                header: "Last run and trend",
                cell: (suite) => <SuiteTrend name={suite.name} />,
              },
              {
                header: "Saved",
                cell: (suite) => formatTime(suite.created_at),
              },
            ]}
          />
        )}
      </Section>
      <Section
        title="New suite"
        description="Checks on a target: thresholds, regressions against a baseline, stage assertions, golden queries. Saving the same suite again adds no version."
      >
        <SuiteEditor initial={EXAMPLE_SUITE} />
      </Section>
      <Section title="Recent test runs" description="Newest first.">
        {runs.data === undefined ? (
          runs.error ? (
            <ErrorNote error={runs.error} />
          ) : (
            <Loading />
          )
        ) : runs.data.items.length === 0 ? (
          <Empty>No test runs yet.</Empty>
        ) : (
          <RunsTable runs={runs.data.items} label="Recent test runs" />
        )}
      </Section>
    </div>
  );
}

function trendOption(runs: HistoryRun[], checkId: string) {
  const points = checkTrend(runs, checkId);
  return {
    animation: false,
    grid: { left: 48, right: 16, top: 16, bottom: 32 },
    tooltip: { trigger: "axis" },
    xAxis: {
      type: "category",
      data: points.map((point) => formatTime(point.at)),
    },
    yAxis: { type: "value", scale: true },
    series: [
      {
        name: checkId,
        type: "line",
        connectNulls: false,
        data: points.map((point) => ({
          value: point.value,
          itemStyle: {
            color: point.result === "pass" ? "#16a34a" : "#dc2626",
          },
        })),
      },
    ],
  };
}

function Trends({ name }: { name: string }) {
  const { api } = useLab();
  const history = useLoad(
    async () =>
      unwrap(
        await api.GET("/test-suites/{name}/history", {
          params: { path: { name }, query: { limit: 50 } },
        }),
      ),
    `trends:${name}`,
  );
  const runs = (history.data?.runs ?? []) as unknown as HistoryRun[];
  const ids = checkIds(runs);
  const [chosen, setChosen] = useState<string | null>(null);
  const checkId = chosen ?? ids[0] ?? null;
  if (history.data === undefined)
    return history.error ? <ErrorNote error={history.error} /> : <Loading />;
  if (runs.length === 0 || checkId === null)
    return <Empty>No finished runs yet.</Empty>;
  return (
    <div className="grid gap-2">
      <label className="grid gap-1 text-xs font-medium">
        Check
        <select
          className={inputClass}
          value={checkId}
          onChange={(event) => setChosen(event.target.value)}
        >
          {ids.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
      </label>
      <Chart
        option={trendOption(runs, checkId)}
        label={`${checkId} across the suite's runs`}
        testId="check-trend"
        renderer="svg"
        height={220}
      />
    </div>
  );
}

function AutomationOf({ name }: { name: string }) {
  const { api } = useLab();
  const automation = useLoad(
    async () =>
      unwrap(
        await api.GET("/test-suites/{name}/automation", {
          params: { path: { name } },
        }),
      ),
    `automation:${name}`,
  );
  if (automation.data === undefined)
    return automation.error ? (
      <ErrorNote error={automation.error} />
    ) : (
      <Loading />
    );
  const { events, schedules, time_zone: zone } = automation.data;
  return (
    <div className="grid gap-2 text-sm" data-testid="suite-automation">
      <p>
        Runs on:{" "}
        {events.length > 0
          ? events.map((e) => e.replace(/_/g, " ")).join(", ")
          : "no events"}
        {schedules.length > 0
          ? `; ${schedules.length} schedule(s), ${zone}`
          : ""}
        .
      </p>
      {schedules.length > 0 ? (
        <DataTable<ScheduleOut>
          label="Schedules"
          rows={schedules}
          rowKey={(schedule) => schedule.cron}
          columns={[
            { header: "Schedule", cell: (s) => <code>{s.cron}</code> },
            { header: "Time zone", cell: (s) => s.time_zone },
            { header: "Next", cell: (s) => formatTime(s.next_fire_at) },
            { header: "Last", cell: (s) => formatTime(s.last_fired_at) },
            {
              header: "Last run",
              cell: (s) =>
                s.last_test_run_id ? (
                  <Link
                    to={{ name: "test-run", testRunId: s.last_test_run_id }}
                    className="text-link font-mono"
                  >
                    {s.last_test_run_id.slice(0, 8)}
                  </Link>
                ) : (
                  "–"
                ),
            },
          ]}
        />
      ) : null}
    </div>
  );
}

/** One suite: its latest version to edit and run, its automation, runs and trends. */
export function SuiteDetail({ suiteName }: { suiteName: string }) {
  const { api } = useLab();
  const navigate = useNavigate();
  const suite = useLoad(
    async () =>
      unwrap(
        await api.GET("/test-suites/{name}", {
          params: { path: { name: suiteName } },
        }),
      ),
    `suite:${suiteName}`,
  );
  const runs = useLoad(
    async () =>
      unwrap(
        await api.GET("/test-runs", {
          params: { query: { suite: suiteName, limit: 50 } },
        }),
      ),
    `suite-runs:${suiteName}`,
  );
  const [error, setError] = useState<unknown>(null);
  const runNow = async () => {
    setError(null);
    try {
      const started = unwrap(
        await api.POST("/test-runs", {
          body: { suite: suiteName, trigger: "api" },
        }),
      );
      navigate({ name: "test-run", testRunId: started.id });
    } catch (failure) {
      setError(failure);
    }
  };
  if (suite.data === undefined)
    return suite.error ? <ErrorNote error={suite.error} /> : <Loading />;
  const stored = suite.data;
  return (
    <div className="grid gap-4">
      <Section
        title={`Suite ${stored.name}`}
        description={`Version ${stored.version}, ${stored.checks} checks, saved ${formatTime(stored.created_at)}.`}
      >
        <div className="flex flex-wrap gap-2">
          <Button size="sm" onPress={() => void runNow()}>
            Run now
          </Button>
        </div>
        <ErrorNote error={error} />
        <div className="mt-3">
          <AutomationOf name={stored.name} />
        </div>
      </Section>
      <Section
        title="Trends"
        description="A check's value in each finished run; red where it didn't pass."
      >
        <Trends name={stored.name} />
      </Section>
      <Section title="Runs" description="Newest first.">
        {runs.data === undefined ? (
          <Loading />
        ) : runs.data.items.length === 0 ? (
          <Empty>This suite hasn't run yet.</Empty>
        ) : (
          <RunsTable runs={runs.data.items} label="Runs of this suite" />
        )}
      </Section>
      <Section
        title="Edit"
        description="The latest version as stored, every default filled in. JSON is YAML too: edit it and save a new version."
      >
        <SuiteEditor
          key={stored.id}
          initial={JSON.stringify(stored.document, null, 2)}
          label="Suite (YAML or JSON)"
        />
      </Section>
    </div>
  );
}

function Evidence({ links }: { links: EvidenceLink[] }) {
  if (links.length === 0) return <span className="text-muted">–</span>;
  return (
    <span className="flex flex-wrap gap-1">
      {links.slice(0, 6).map((link, index) => {
        const route = evidenceRoute(link);
        const text = `${link.kind} ${link.id.length > 12 ? link.id.slice(0, 8) : link.id}`;
        return route ? (
          <Link key={index} to={route} className="text-link text-xs">
            {text}
          </Link>
        ) : (
          <span key={index} className="text-xs">
            {text}
          </span>
        );
      })}
    </span>
  );
}

const REPORTS = [
  {
    format: "json",
    label: "JSON record",
    type: "application/json",
    ext: "json",
  },
  { format: "junit", label: "JUnit XML", type: "application/xml", ext: "xml" },
  { format: "markdown", label: "Markdown", type: "text/markdown", ext: "md" },
] as const;

/** A test run: its inputs, each check's result with its evidence, and its reports. */
export function TestRunDetail({ testRunId }: { testRunId: string }) {
  const { api, host } = useLab();
  const run = useLiveTestRun(testRunId);
  const [error, setError] = useState<unknown>(null);
  if (run.data === undefined)
    return run.error ? <ErrorNote error={run.error} /> : <Loading />;
  const found = run.data;
  const finished = ["passed", "failed", "error", "cancelled"].includes(
    found.status,
  );
  const resolved = found.resolved as Record<
    string,
    Record<string, unknown> | null
  >;
  const target = resolved.target ?? {};
  const named = (value: unknown) => {
    const item = (value ?? {}) as Record<string, unknown>;
    return item.name ? `${String(item.name)} v${String(item.version)}` : "–";
  };

  const download = async (format: (typeof REPORTS)[number]) => {
    setError(null);
    try {
      const text = unwrap(
        await api.GET("/test-runs/{test_run_id}/report", {
          params: {
            path: { test_run_id: testRunId },
            query: { format: format.format },
          },
          parseAs: "text",
        }),
      ) as unknown as string;
      downloadText(
        text,
        `test-run-${testRunId.slice(0, 8)}.${format.ext}`,
        format.type,
      );
    } catch (failure) {
      setError(failure);
    }
  };

  const cancel = async () => {
    setError(null);
    try {
      unwrap(
        await api.POST("/test-runs/{test_run_id}/cancel", {
          params: { path: { test_run_id: testRunId } },
        }),
      );
      host.notify("The test run is cancelled.", "success");
      run.reload();
    } catch (failure) {
      setError(failure);
    }
  };

  return (
    <div className="grid gap-4">
      <Section
        title={`Test run ${testRunId.slice(0, 8)}`}
        description={`Suite ${found.suite_name} v${found.suite_version}, from ${found.triggers.join(", ")}.`}
      >
        <div className="grid gap-2 text-sm" data-testid="test-run-status">
          <div className="flex flex-wrap items-center gap-2">
            <StatusChip status={found.status} />
            <span>{runSummary(found)}</span>
            {found.error_code ? (
              <span className="text-danger">
                ({found.error_code.replace(/_/g, " ")})
              </span>
            ) : null}
          </div>
          <p>
            <Link
              to={{ name: "suite", suiteName: found.suite_name }}
              className="text-link"
            >
              The suite
            </Link>
            . Target: {String(target.kind ?? "–")}{" "}
            {target.profile ? named(target.profile) : named(target.pipeline)}
            {target.retrieval ? `, ${named(target.retrieval)}` : ""}. Corpus{" "}
            {named(resolved.corpus)}
            {resolved.eval_set
              ? `; eval set ${named(resolved.eval_set)} (dev)`
              : ""}
            {resolved.baseline
              ? `; baseline ${String(resolved.baseline.kind)}${
                  resolved.baseline.found ? "" : " (none yet)"
                }`
              : ""}
            .
          </p>
          <p className="flex flex-wrap gap-3">
            {found.pipeline_run_id ? (
              <Link
                to={{ name: "run", runId: found.pipeline_run_id }}
                className="text-link"
              >
                Pipeline run
              </Link>
            ) : null}
            {found.eval_run_id ? (
              <Link
                to={{ name: "eval", evalRunId: found.eval_run_id }}
                className="text-link"
              >
                Eval run
              </Link>
            ) : null}
            {found.baseline_eval_run_id && found.eval_run_id ? (
              <Link
                to={{
                  name: "eval-compare",
                  a: found.baseline_eval_run_id,
                  b: found.eval_run_id,
                }}
                className="text-link"
              >
                Compare with the baseline
              </Link>
            ) : null}
          </p>
          <div className="flex flex-wrap gap-2">
            {finished ? (
              REPORTS.map((format) => (
                <Button
                  key={format.format}
                  size="sm"
                  variant="secondary"
                  onPress={() => void download(format)}
                >
                  {format.label}
                </Button>
              ))
            ) : (
              <Button
                size="sm"
                variant="secondary"
                onPress={() => void cancel()}
              >
                Cancel
              </Button>
            )}
          </div>
          <ErrorNote error={error} />
        </div>
      </Section>
      {(found.code ?? []).length > 0 ? (
        <Section
          title="Custom check code"
          description="The files the suite's custom checks ran, by SHA-256: the code is kept with the suite's version."
        >
          <DataTable<CodeFileOut>
            label="Custom check code"
            rows={found.code ?? []}
            rowKey={(file) => file.path}
            columns={[
              { header: "File", cell: (file) => <code>{file.path}</code> },
              {
                header: "SHA-256",
                cell: (file) => (
                  <code className="break-all">{file.sha256}</code>
                ),
              },
              { header: "Bytes", cell: (file) => file.bytes },
            ]}
          />
        </Section>
      ) : null}
      <Section
        title="Checks"
        description="Each check's value, what it was held to, its baseline, and where its evidence is."
      >
        {!finished ? (
          <Loading label={`The run is ${found.status}`} />
        ) : (found.checks ?? []).length === 0 ? (
          <Empty>No checks were evaluated.</Empty>
        ) : (
          <DataTable<CheckOut>
            label="Checks"
            rows={found.checks ?? []}
            rowKey={(check) => check.id}
            columns={[
              { header: "Check", cell: (check) => <code>{check.id}</code> },
              {
                header: "Type",
                cell: (check) => check.type.replace(/_/g, " "),
              },
              {
                header: "Result",
                cell: (check) => <StatusChip status={check.result} />,
              },
              {
                header: "Value",
                cell: (check) =>
                  check.value === null
                    ? "–"
                    : Number(check.value.toPrecision(4)),
              },
              {
                header: "Threshold",
                cell: (check) => thresholdText(check.threshold) || "–",
              },
              {
                header: "Baseline",
                cell: (check) =>
                  check.baseline_value === null
                    ? "–"
                    : Number(check.baseline_value.toPrecision(4)),
              },
              {
                header: "Why",
                cell: (check) =>
                  check.code ? check.code.replace(/_/g, " ") : "",
              },
              {
                header: "Evidence",
                cell: (check) => (
                  <Evidence
                    links={check.evidence as unknown as EvidenceLink[]}
                  />
                ),
              },
            ]}
          />
        )}
      </Section>
    </div>
  );
}

function OwnersOnly({ error }: { error: unknown }) {
  if (error instanceof ApiError && error.status === 403)
    return <Empty>Owners manage these.</Empty>;
  return <ErrorNote error={error} />;
}

function Tokens() {
  const { api, host } = useLab();
  const [tick, setTick] = useState(0);
  const tokens = useLoad(
    async () => unwrap(await api.GET("/api-tokens")),
    `tokens:${tick}`,
  );
  const [name, setName] = useState("ci-main");
  const [scopes, setScopes] = useState<string[]>(["read", "tests:run"]);
  const [days, setDays] = useState(90);
  const [created, setCreated] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);

  const create = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    try {
      const made = unwrap(
        await api.POST("/api-tokens", {
          body: {
            name,
            scopes: scopes as ("read" | "runs:write" | "tests:run")[],
            expires_in_days: days,
          },
        }),
      );
      setCreated(made.token);
      setTick((value) => value + 1);
    } catch (failure) {
      setError(failure);
    }
  };

  const revoke = async (token: TokenOut) => {
    const sure = await host.confirm({
      title: `Revoke ${token.name}?`,
      body: "Whatever uses it stops working at once.",
      confirmLabel: "Revoke",
    });
    if (!sure) return;
    try {
      unwrap(
        await api.DELETE("/api-tokens/{token_id}", {
          params: { path: { token_id: token.id } },
        }),
      );
      setTick((value) => value + 1);
    } catch (failure) {
      setError(failure);
    }
  };

  if (tokens.error) return <OwnersOnly error={tokens.error} />;
  return (
    <div className="grid gap-3">
      <form
        onSubmit={create}
        className="flex flex-wrap items-end gap-2"
        aria-label="New API token"
      >
        <label className="grid gap-1 text-xs font-medium">
          Token name
          <input
            className={inputClass}
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </label>
        <fieldset className="flex gap-2 text-xs">
          <legend className="font-medium">Scopes</legend>
          {SCOPES.map((scope) => (
            <label key={scope} className="flex items-center gap-1">
              <input
                type="checkbox"
                checked={scopes.includes(scope)}
                onChange={(event) =>
                  setScopes(
                    event.target.checked
                      ? [...scopes, scope]
                      : scopes.filter((item) => item !== scope),
                  )
                }
              />
              {scope}
            </label>
          ))}
        </fieldset>
        <label className="grid gap-1 text-xs font-medium">
          Days
          <input
            type="number"
            min={1}
            max={365}
            className={`${inputClass} w-20`}
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
          />
        </label>
        <Button type="submit" size="sm" isDisabled={scopes.length === 0}>
          Make token
        </Button>
      </form>
      <ErrorNote error={error} />
      {created ? (
        <div
          className="grid gap-1 text-sm"
          role="status"
          data-testid="new-token"
        >
          <p>Copy it now: it isn't shown again.</p>
          <code className="break-all rounded border border-separator p-2">
            {created}
          </code>
        </div>
      ) : null}
      {tokens.data === undefined ? (
        <Loading />
      ) : tokens.data.items.length === 0 ? (
        <Empty>No API tokens yet.</Empty>
      ) : (
        <DataTable<TokenOut>
          label="API tokens"
          rows={tokens.data.items}
          rowKey={(token) => token.id}
          columns={[
            { header: "Name", cell: (t) => t.name },
            { header: "Starts", cell: (t) => <code>{t.prefix}…</code> },
            { header: "Scopes", cell: (t) => t.scopes.join(", ") },
            {
              header: "State",
              cell: (t) => (
                <StatusChip
                  status={t.active ? "active" : "cancelled"}
                  label={t.active ? "active" : "revoked or expired"}
                />
              ),
            },
            { header: "Expires", cell: (t) => formatTime(t.expires_at) },
            { header: "Last used", cell: (t) => formatTime(t.last_used_at) },
            {
              header: "",
              cell: (t) =>
                t.active ? (
                  <Button
                    size="sm"
                    variant="secondary"
                    onPress={() => void revoke(t)}
                  >
                    Revoke
                  </Button>
                ) : null,
            },
          ]}
        />
      )}
    </div>
  );
}

function Webhooks() {
  const { api } = useLab();
  const [tick, setTick] = useState(0);
  const webhooks = useLoad(
    async () => unwrap(await api.GET("/webhooks")),
    `webhooks:${tick}`,
  );
  const [url, setUrl] = useState("");
  const [suite, setSuite] = useState("");
  const [secret, setSecret] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);

  const create = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    try {
      const made = unwrap(
        await api.POST("/webhooks", {
          body: { url, suite: suite || null, events: ["test_run.finished"] },
        }),
      );
      setSecret(made.secret);
      setTick((value) => value + 1);
    } catch (failure) {
      setError(failure);
    }
  };

  const remove = async (webhook: WebhookOut) => {
    try {
      await api.DELETE("/webhooks/{webhook_id}", {
        params: { path: { webhook_id: webhook.id } },
      });
      setTick((value) => value + 1);
    } catch (failure) {
      setError(failure);
    }
  };

  if (webhooks.error) return <OwnersOnly error={webhooks.error} />;
  return (
    <div className="grid gap-3">
      <form
        onSubmit={create}
        className="flex flex-wrap items-end gap-2"
        aria-label="New webhook"
      >
        <label className="grid gap-1 text-xs font-medium">
          Webhook URL
          <input
            className={`${inputClass} w-80`}
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
        </label>
        <label className="grid gap-1 text-xs font-medium">
          Only this suite (optional)
          <input
            className={inputClass}
            value={suite}
            onChange={(e) => setSuite(e.target.value)}
          />
        </label>
        <Button type="submit" size="sm" isDisabled={!url}>
          Add webhook
        </Button>
      </form>
      <ErrorNote error={error} />
      {secret ? (
        <div className="grid gap-1 text-sm" role="status">
          <p>
            Its signing secret, shown this once: deliveries carry
            Stratafi-Signature.
          </p>
          <code className="break-all rounded border border-separator p-2">
            {secret}
          </code>
        </div>
      ) : null}
      {webhooks.data === undefined ? (
        <Loading />
      ) : webhooks.data.items.length === 0 ? (
        <Empty>No webhooks yet.</Empty>
      ) : (
        <DataTable<WebhookOut>
          label="Webhooks"
          rows={webhooks.data.items}
          rowKey={(webhook) => webhook.id}
          columns={[
            {
              header: "URL",
              cell: (w) => <code className="break-all">{w.url}</code>,
            },
            { header: "Suite", cell: (w) => w.suite ?? "every suite" },
            { header: "Events", cell: (w) => w.events.join(", ") },
            { header: "Added", cell: (w) => formatTime(w.created_at) },
            {
              header: "",
              cell: (w) => (
                <Button
                  size="sm"
                  variant="secondary"
                  onPress={() => void remove(w)}
                >
                  Remove
                </Button>
              ),
            },
          ]}
        />
      )}
    </div>
  );
}

const BLOCKED: Record<string, string> = {
  sandbox_not_configured:
    "this deployment has no custom-check sandbox (suites.sandbox)",
  sandbox_not_reviewed:
    "the sandbox's security review isn't signed off (suites.sandbox.review)",
};

/** The workspace's custom checks: off until an owner turns them on, after the review. */
function CustomChecks() {
  const { api } = useLab();
  const [tick, setTick] = useState(0);
  const features = useLoad(
    async () => unwrap(await api.GET("/workspace/features")),
    `features:${tick}`,
  );
  const [error, setError] = useState<unknown>(null);
  if (features.error) return <ErrorNote error={features.error} />;
  if (features.data === undefined) return <Loading />;
  const feature = features.data.items.find(
    (item) => item.name === "custom_checks",
  );
  if (feature === undefined) return null;

  const change = async (enabled: boolean) => {
    setError(null);
    try {
      unwrap(
        await api.PUT("/workspace/features/{feature}", {
          params: { path: { feature: "custom_checks" } },
          body: { enabled },
        }),
      );
      setTick((value) => value + 1);
    } catch (failure) {
      setError(failure);
    }
  };

  return (
    <div className="grid gap-2 text-sm" data-testid="custom-checks">
      <p className="flex flex-wrap items-center gap-2">
        <StatusChip
          status={feature.enabled ? "active" : "cancelled"}
          label={feature.enabled ? "on" : "off"}
        />
        {feature.enabled
          ? `Custom checks run in the sandbox, under security review ${feature.review ?? "–"}; ${feature.changed_by ?? "an owner"} turned them on ${formatTime(feature.changed_at)}.`
          : feature.blocked
            ? `They can't be turned on yet: ${BLOCKED[feature.blocked] ?? feature.blocked.replace(/_/g, " ")}. Until then each custom check ends as an error, custom checks disabled.`
            : "Each custom check ends as an error, custom checks disabled, until an owner turns them on."}
      </p>
      <div>
        <Button
          size="sm"
          variant="secondary"
          isDisabled={!feature.enabled && feature.blocked !== null}
          onPress={() => void change(!feature.enabled)}
        >
          {feature.enabled ? "Turn custom checks off" : "Turn custom checks on"}
        </Button>
      </div>
      <ErrorNote error={error} />
    </div>
  );
}

/** What runs suites on their own, and what scripts and CI sign in with. */
export function Automation() {
  const { api } = useLab();
  const suites = useLoad(
    async () =>
      unwrap(
        await api.GET("/test-suites", { params: { query: { limit: 100 } } }),
      ),
    "automation-suites",
  );
  return (
    <div className="grid gap-4">
      <Section
        title="Schedules and triggers"
        description="A suite runs when its corpus gets a new version, its target changes, or a schedule comes due; its YAML's triggers say which."
      >
        {suites.data === undefined ? (
          <Loading />
        ) : suites.data.items.length === 0 ? (
          <Empty>No suites yet.</Empty>
        ) : (
          <div className="grid gap-3">
            {suites.data.items.map((suite) => (
              <div key={suite.id} className="grid gap-1">
                <Link
                  to={{ name: "suite", suiteName: suite.name }}
                  className="text-link text-sm font-medium"
                >
                  {suite.name}
                </Link>
                <AutomationOf name={suite.name} />
              </div>
            ))}
          </div>
        )}
      </Section>
      <Section
        title="API tokens"
        description="For scripts and CI (examples/ci/github-actions.yml): read, runs:write, tests:run. Only a token's hash is kept; it expires, and revoking it is immediate."
      >
        <Tokens />
      </Section>
      <Section
        title="Webhooks"
        description="Where finished test runs are announced, signed with the webhook's secret and retried with backoff."
      >
        <Webhooks />
      </Section>
      <Section
        title="Custom checks"
        description="Python functions in a suite (python: checks/file.py:function), run in a sandbox with no network and no credentials. Owners turn them on once the sandbox's security review is signed off."
      >
        <CustomChecks />
      </Section>
    </div>
  );
}
