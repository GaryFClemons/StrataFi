import { Button } from "@heroui/react";
import { Suspense, lazy, useState, type FormEvent } from "react";
import type { components } from "../api";
import { unwrap, useLoad } from "../api/load";
import { useLiveEvalRun } from "../components/live";
import { TrackingTable } from "../components/TrackingTable";
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
import {
  CHECKPOINTS,
  CHECKPOINT_LABELS,
  LOST_LABELS,
  RULES,
  estimateOf,
  formatEstimate,
  highlightSegments,
  type Checkpoint,
  type Rule,
} from "../evals";
import { formatDuration, formatTime, isActive, shortHash } from "../format";
import { AnswerRuns } from "./Answers";

const EvalRunPanels = lazy(() =>
  import("./Panels").then((module) => ({ default: module.EvalRunPanels })),
);
const QueryTrace = lazy(() =>
  import("./Panels").then((module) => ({ default: module.QueryTrace })),
);

type EvalRunOut = components["schemas"]["EvalRunOut"];
type QueryResultOut = components["schemas"]["QueryResultOut"];
type PassageOut = components["schemas"]["PassageOut"];
type Split = components["schemas"]["EvalRunIn"]["split"];
type Outcome = "" | "hit" | "miss" | "abstain";

const HYBRID = {
  schema: "stratafi.retrieval.v1" as const,
  name: "hybrid-v1",
  description:
    "Hybrid search: BM25 and exact dense, RRF k = 60 over 50, no reranker, top 8",
  retrievers: { dense: { top_n: 50 }, keyword: { top_n: 50 } },
  fusion: { method: "rrf" as const, k: 60, alpha: 0.5 },
  candidates: 50,
  filters: { current_only: true },
  rerank: { component: "rerank/none" },
  top_k: 8,
};

function refLabel(ref: Record<string, unknown> | null | undefined): string {
  if (!ref) return "–";
  return `${String(ref.name ?? "?")} v${String(ref.version ?? "?")}`;
}

function kEval(run: EvalRunOut): number {
  return run.k_eval ?? 5;
}

/** Start an eval run, and the list of them. */
export function Evals() {
  const { api, host } = useLab();
  const navigate = useNavigate();
  const runs = useLoad(
    async () =>
      unwrap(
        await api.GET("/runs", {
          params: { query: { limit: 50, status: "succeeded" } },
        }),
      ),
    "runs",
  );
  const configs = useLoad(
    async () =>
      unwrap(
        await api.GET("/retrieval-configs", {
          params: { query: { limit: 100 } },
        }),
      ),
    "configs",
  );
  const sets = useLoad(
    async () =>
      unwrap(
        await api.GET("/eval-sets", { params: { query: { limit: 100 } } }),
      ),
    "eval-sets",
  );
  const [runId, setRunId] = useState("");
  const [configId, setConfigId] = useState("");
  const [versionId, setVersionId] = useState("");
  const [split, setSplit] = useState<Split>("dev");
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const frozen = (sets.data?.items ?? []).filter(
    (set) => set.latest_version?.frozen,
  );

  const start = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const started = unwrap(
        await api.POST("/eval-runs", {
          body: {
            run_id: runId || (runs.data?.items[0]?.id ?? ""),
            retrieval_config_id: configId || (configs.data?.items[0]?.id ?? ""),
            eval_set_version_id:
              versionId || (frozen[0]?.latest_version?.id ?? ""),
            split,
            confirm_test_read: confirm,
            options: {
              shard_size: 25,
              coverage_threshold: 0.7,
              resamples: 2000,
              seed: 0,
            },
          },
        }),
      );
      navigate({ name: "eval", evalRunId: started.id });
    } catch (failure) {
      setError(failure);
    } finally {
      setBusy(false);
    }
  };

  const addDefault = async () => {
    setError(null);
    try {
      unwrap(await api.POST("/retrieval-configs", { body: HYBRID }));
      host.notify("hybrid-v1 saved.", "success");
      configs.reload();
    } catch (failure) {
      setError(failure);
    }
  };

  return (
    <div className="grid gap-4">
      <Section
        title="Evaluate"
        description="Search a run's indexes with a retrieval config for every question in an eval set, and score where the evidence was found or lost."
      >
        <form
          onSubmit={start}
          aria-label="Start an eval run"
          className="flex flex-wrap items-end gap-2"
        >
          <label className="grid gap-1 text-xs font-medium">
            Pipeline run
            <select
              className={inputClass}
              value={runId}
              onChange={(event) => setRunId(event.target.value)}
            >
              {(runs.data?.items ?? []).map((run) => (
                <option key={run.id} value={run.id}>
                  {shortHash(run.id, 8)} · {formatTime(run.created_at)}
                </option>
              ))}
            </select>
          </label>
          <label className="grid gap-1 text-xs font-medium">
            Retrieval config
            <select
              className={inputClass}
              value={configId}
              onChange={(event) => setConfigId(event.target.value)}
            >
              {(configs.data?.items ?? []).map((config) => (
                <option key={config.id} value={config.id}>
                  {config.name} v{config.version}
                </option>
              ))}
            </select>
          </label>
          <label className="grid gap-1 text-xs font-medium">
            Eval set
            <select
              className={inputClass}
              value={versionId}
              onChange={(event) => setVersionId(event.target.value)}
            >
              {frozen.map((set) => (
                <option key={set.id} value={set.latest_version?.id ?? ""}>
                  {set.name} v{set.latest_version?.version}
                </option>
              ))}
            </select>
          </label>
          <label className="grid gap-1 text-xs font-medium">
            Split
            <select
              className={inputClass}
              value={split}
              onChange={(event) => setSplit(event.target.value as Split)}
            >
              <option value="dev">dev</option>
              <option value="test">test (locked)</option>
              <option value="all">all (reads test)</option>
            </select>
          </label>
          {split === "dev" ? null : (
            <label className="flex items-center gap-1 text-xs font-medium">
              <input
                type="checkbox"
                checked={confirm}
                onChange={(event) => setConfirm(event.target.checked)}
              />
              Confirm a counted test-split read
            </label>
          )}
          <Button
            type="submit"
            size="sm"
            isDisabled={
              busy ||
              !runs.data?.items.length ||
              !configs.data?.items.length ||
              frozen.length === 0
            }
          >
            Evaluate
          </Button>
          {configs.data && configs.data.items.length === 0 ? (
            <Button size="sm" variant="secondary" onPress={addDefault}>
              Add hybrid-v1
            </Button>
          ) : null}
        </form>
        <ErrorNote error={error} />
      </Section>
      <TrackingTable />
    </div>
  );
}

function MetricCard({
  label,
  metrics,
  rule,
  checkpoint,
  name,
}: {
  label: string;
  metrics: Record<string, unknown> | null | undefined;
  rule: Rule;
  checkpoint: Checkpoint;
  name: string;
}) {
  const estimate = estimateOf(metrics, rule, checkpoint, name);
  return (
    <div
      className="grid gap-1 rounded-md border border-separator p-3"
      data-testid="metric-card"
    >
      <span className="text-xs text-muted">{label}</span>
      <span className="text-2xl font-semibold">
        {estimate ? estimate.mean.toFixed(3) : "–"}
      </span>
      <span className="text-xs text-muted" data-testid="metric-ci">
        {estimate
          ? `95% CI ${estimate.low.toFixed(3)}–${estimate.high.toFixed(3)}, n = ${estimate.n}`
          : "no data"}
      </span>
    </div>
  );
}

function Count({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="grid gap-0.5 text-sm">
      <span className="text-xs text-muted">{label}</span>
      <span>{value}</span>
    </div>
  );
}

function sliceRows(
  slices: Record<string, Record<string, unknown>> | undefined,
): { label: string; summary: Record<string, unknown> }[] {
  return Object.entries(slices ?? {})
    .filter(([, summary]) => Number(summary.with_evidence ?? 0) > 0)
    .map(([label, summary]) => ({ label, summary }));
}

export function EvalResults({ evalRunId }: { evalRunId: string }) {
  const { api } = useLab();
  const live = useLiveEvalRun(evalRunId);
  const [rule, setRule] = useState<Rule>("knowledge-hit-v1");
  const [checkpoint, setCheckpoint] = useState<Checkpoint>("final");
  const [outcome, setOutcome] = useState<Outcome>("");
  const run = live.data;
  const finished = run !== undefined && !isActive(run.status);
  const queries = useLoad(
    async () =>
      finished
        ? unwrap(
            await api.GET("/eval-runs/{eval_run_id}/queries", {
              params: {
                path: { eval_run_id: evalRunId },
                query: { limit: 200, ...(outcome ? { outcome } : {}) },
              },
            }),
          )
        : null,
    `${evalRunId}:${finished}:${outcome}`,
  );
  if (run === undefined) {
    return live.error ? <ErrorNote error={live.error} /> : <Loading />;
  }
  const k = kEval(run);
  const metrics = run.metrics as Record<string, unknown> | null;
  const shards = (run.counts.shards ?? {}) as Record<string, number>;
  const done = (shards.succeeded ?? 0) + (shards.cached ?? 0);
  const total = Object.values(shards).reduce((sum, count) => sum + count, 0);
  const lost = (metrics?.where_lost ?? {}) as Record<string, number>;
  const violations = (metrics?.violations ?? {}) as Record<
    string,
    { count: number; of: number }
  >;
  const findability = (metrics?.findability ?? {}) as {
    findable?: number;
    of?: number;
    rate?: number | null;
  };
  const uplift = (
    (metrics?.uplift ?? {}) as Record<string, Record<string, unknown>>
  )[rule]?.[`recall@${k}`] as
    { mean: number; low: number; high: number; n: number } | undefined;
  const latency = (metrics?.latency_ms ?? {}) as Record<
    string,
    { p50: number | null; p95: number | null }
  >;
  const slices = (metrics?.slices ?? {}) as Record<
    string,
    Record<string, Record<string, unknown>>
  >;

  return (
    <div className="grid gap-4">
      <p className="text-sm">
        <Link to={{ name: "evals" }} className="text-link">
          Evals
        </Link>{" "}
        / <code>{shortHash(run.id, 8)}</code>
      </p>
      <Section
        title="Eval run"
        description={`${refLabel(run.eval_set)} (${run.split}) · ${refLabel(run.retrieval_ref)} · k = ${k}`}
        actions={<StatusChip status={run.status} />}
      >
        <div className="flex flex-wrap gap-6" data-testid="eval-progress">
          <Count label="Shards" value={`${done} of ${total}`} />
          <Count
            label="Queries"
            value={Number(run.counts.queries ?? 0) || "–"}
          />
          <Count
            label="Reused results"
            value={run.results_cached ? "yes" : "no"}
          />
          <Count
            label="Test split read"
            value={run.test_read ? "yes (counted)" : "no"}
          />
          <Count
            label="Latency p50 / p95"
            value={`${formatDuration(latency.total_ms?.p50)} / ${formatDuration(latency.total_ms?.p95)}`}
          />
          {run.error_code ? (
            <Count label="Error" value={run.error_code} />
          ) : null}
        </div>
      </Section>
      {run.status !== "succeeded" ? null : (
        <>
          <Section
            title="Metrics"
            description="Means over the queries with evidence, each with its 95% bootstrap interval."
            actions={
              <>
                <label className="flex items-center gap-1 text-xs font-medium">
                  Rule
                  <select
                    className={inputClass}
                    value={rule}
                    onChange={(event) => setRule(event.target.value as Rule)}
                  >
                    {RULES.map((item) => (
                      <option key={item} value={item}>
                        {item}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="flex items-center gap-1 text-xs font-medium">
                  Checkpoint
                  <select
                    className={inputClass}
                    value={checkpoint}
                    onChange={(event) =>
                      setCheckpoint(event.target.value as Checkpoint)
                    }
                  >
                    {CHECKPOINTS.map((item) => (
                      <option key={item} value={item}>
                        {CHECKPOINT_LABELS[item]}
                      </option>
                    ))}
                  </select>
                </label>
              </>
            }
          >
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <MetricCard
                label={`hit@${k}`}
                metrics={metrics}
                rule={rule}
                checkpoint={checkpoint}
                name={`hit@${k}`}
              />
              <MetricCard
                label={`recall@${k}`}
                metrics={metrics}
                rule={rule}
                checkpoint={checkpoint}
                name={`recall@${k}`}
              />
              <MetricCard
                label="MRR@10"
                metrics={metrics}
                rule={rule}
                checkpoint={checkpoint}
                name="mrr@10"
              />
              <MetricCard
                label="nDCG@10"
                metrics={metrics}
                rule={rule}
                checkpoint={checkpoint}
                name="ndcg@10"
              />
            </div>
          </Section>
          <Section
            title="Checkpoints"
            description="Where evidence survives along the way: each retriever, fusion, the candidate pool the reranker sees, and the final list."
          >
            <DataTable<Checkpoint>
              label="Checkpoints"
              rows={[...CHECKPOINTS]}
              rowKey={(item) => item}
              columns={[
                {
                  header: "Checkpoint",
                  cell: (item) => CHECKPOINT_LABELS[item],
                },
                {
                  header: `recall@${k}`,
                  cell: (item) =>
                    formatEstimate(
                      estimateOf(metrics, rule, item, `recall@${k}`),
                    ),
                },
                {
                  header: `hit@${k}`,
                  cell: (item) =>
                    formatEstimate(estimateOf(metrics, rule, item, `hit@${k}`)),
                },
                {
                  header: "recall@all",
                  cell: (item) =>
                    formatEstimate(
                      estimateOf(metrics, rule, item, "recall@all"),
                    ),
                },
                {
                  header: "MRR@10",
                  cell: (item) =>
                    formatEstimate(estimateOf(metrics, rule, item, "mrr@10")),
                },
              ]}
            />
            <div className="mt-3 flex flex-wrap gap-6">
              <Count
                label="Rerank uplift, recall (final − fused)"
                value={formatEstimate(uplift)}
              />
              <Count
                label="Found / lost at parse / retrieval / rerank"
                value={`${lost.found ?? 0} / ${lost.parse ?? 0} / ${lost.retrieval ?? 0} / ${lost.rerank ?? 0}`}
              />
              <Count
                label="Misses the filter hid (as-of date, ACL, status)"
                value={lost.filtered ?? 0}
              />
              <Count
                label="avoid violations"
                value={`${violations.avoid?.count ?? 0} of ${violations.avoid?.of ?? 0}`}
              />
              <Count
                label="must-not-return violations"
                value={`${violations.must_not_return?.count ?? 0} of ${violations.must_not_return?.of ?? 0}`}
              />
              <Count
                label="Evidence findability"
                value={
                  findability.rate === null || findability.rate === undefined
                    ? "–"
                    : `${findability.findable} of ${findability.of}`
                }
              />
            </div>
          </Section>
          {(["tag", "file_type"] as const).map((kind) => (
            <Section
              key={kind}
              title={kind === "tag" ? "By tag" : "By file type"}
              description="Each slice's own intervals; few queries in a slice means a wide interval."
            >
              <DataTable<{ label: string; summary: Record<string, unknown> }>
                label={kind === "tag" ? "Slices by tag" : "Slices by file type"}
                rows={sliceRows(slices[kind])}
                rowKey={(row) => row.label}
                columns={[
                  {
                    header: kind === "tag" ? "Tag" : "File type",
                    cell: (row) => row.label,
                  },
                  {
                    header: "Queries",
                    cell: (row) => String(row.summary.with_evidence ?? 0),
                  },
                  {
                    header: `hit@${k}`,
                    cell: (row) =>
                      formatEstimate(
                        estimateOf(row.summary, rule, "final", `hit@${k}`),
                      ),
                  },
                  {
                    header: `recall@${k}`,
                    cell: (row) =>
                      formatEstimate(
                        estimateOf(row.summary, rule, "final", `recall@${k}`),
                      ),
                  },
                ]}
              />
            </Section>
          ))}
          <Section
            title="Queries"
            description="Each query: did the final list hit at k, and where a miss lost its evidence."
            actions={
              <label className="flex items-center gap-1 text-xs font-medium">
                Show
                <select
                  className={inputClass}
                  aria-label="Show queries"
                  value={outcome}
                  onChange={(event) =>
                    setOutcome(event.target.value as Outcome)
                  }
                >
                  <option value="">all</option>
                  <option value="miss">misses</option>
                  <option value="hit">hits</option>
                  <option value="abstain">no evidence</option>
                </select>
              </label>
            }
          >
            {queries.data === undefined || queries.data === null ? (
              <Loading />
            ) : (
              <DataTable<QueryResultOut>
                label="Queries"
                rows={queries.data.items}
                rowKey={(row) => row.item_id}
                columns={[
                  {
                    header: "Item",
                    cell: (row) => (
                      <Link
                        to={{
                          name: "eval-query",
                          evalRunId,
                          itemId: row.item_id,
                        }}
                        className="text-link"
                      >
                        {row.item_id}
                      </Link>
                    ),
                  },
                  {
                    header: "Outcome",
                    cell: (row) => (
                      <StatusChip
                        status={
                          row.hit === null
                            ? "abstain"
                            : row.hit
                              ? "succeeded"
                              : "failed"
                        }
                        label={
                          row.hit === null
                            ? "no evidence"
                            : row.hit
                              ? "hit"
                              : "miss"
                        }
                      />
                    ),
                  },
                  {
                    header: "First hit",
                    cell: (row) => row.first_hit_rank ?? "–",
                  },
                  { header: "Lost at", cell: (row) => row.lost_at ?? "–" },
                  { header: "Tags", cell: (row) => row.tags.join(", ") },
                  { header: "Split", cell: (row) => row.split },
                  {
                    header: "Violations",
                    cell: (row) =>
                      [
                        row.avoid_violation ? "avoid" : null,
                        row.must_not_return_violation
                          ? "must-not-return"
                          : null,
                      ]
                        .filter(Boolean)
                        .join(", ") || "–",
                  },
                ]}
              />
            )}
          </Section>
          <AnswerRuns evalRunId={evalRunId} />
        </>
      )}
      <Suspense fallback={<Loading label="Loading the run's panels" />}>
        <EvalRunPanels
          evalRunId={evalRunId}
          finished={run.status === "succeeded"}
        />
      </Suspense>
    </div>
  );
}

function Highlighted({ passage }: { passage: PassageOut }) {
  const spans = passage.highlights.map(
    ([start = 0, end = 0]) => [start, end] as [number, number],
  );
  return (
    <pre
      aria-label="Passage text"
      className="max-h-60 overflow-auto whitespace-pre-wrap break-words rounded-md border border-separator p-2 font-mono text-xs"
    >
      {highlightSegments(passage.text, spans).map((segment, index) =>
        segment.marked ? (
          <mark key={index} data-testid="evidence-mark">
            {segment.text}
          </mark>
        ) : (
          <span key={index}>{segment.text}</span>
        ),
      )}
    </pre>
  );
}

export function EvalQuery({
  evalRunId,
  itemId,
}: {
  evalRunId: string;
  itemId: string;
}) {
  const { api } = useLab();
  const detail = useLoad(
    async () =>
      unwrap(
        await api.GET("/eval-runs/{eval_run_id}/queries/{item_id}", {
          params: { path: { eval_run_id: evalRunId, item_id: itemId } },
        }),
      ),
    `${evalRunId}:${itemId}`,
  );
  const [checkpoint, setCheckpoint] = useState<Checkpoint>("final");
  if (detail.data === undefined) {
    return detail.error ? <ErrorNote error={detail.error} /> : <Loading />;
  }
  const { result, item, checkpoints, lengths } = detail.data;
  const score = result.score as {
    found_in?: Record<string, boolean>;
    findable?: (boolean | null)[];
    hidden?: boolean[];
  };
  const filtered =
    result.lost_at === "retrieval" &&
    (score.hidden ?? []).length > 0 &&
    (score.hidden ?? []).every(Boolean);
  const expected = (item.expected ?? []) as Record<string, unknown>[];
  const asker = (item.asker ?? {}) as Record<string, unknown>;
  const passages = checkpoints[checkpoint] ?? [];
  return (
    <div className="grid gap-4">
      <p className="text-sm">
        <Link to={{ name: "evals" }} className="text-link">
          Evals
        </Link>{" "}
        /{" "}
        <Link to={{ name: "eval", evalRunId }} className="text-link">
          <code>{shortHash(evalRunId, 8)}</code>
        </Link>{" "}
        / {itemId}
      </p>
      <Section title={`Query ${itemId}`}>
        <div className="grid gap-3">
          <pre
            aria-label="Query"
            className="whitespace-pre-wrap break-words text-sm"
          >
            {String(item.query ?? "")}
          </pre>
          <p className="text-xs text-muted">
            Asked by {String(asker.name ?? "an unnamed asker")} (
            {((asker.groups ?? []) as string[]).join(", ") || "no groups"})
          </p>
          <div data-testid="lost-at" className="text-sm font-medium">
            {result.has_evidence
              ? result.hit
                ? `Hit: the first relevant passage is at rank ${result.first_hit_rank}.`
                : filtered
                  ? "Missed. Retrieval: the evidence's version isn't visible to this asker on the as-of date, so no retriever could return it."
                  : `Missed. ${LOST_LABELS[result.lost_at ?? ""] ?? "Lost somewhere."}`
              : "No evidence: the answer isn't in the corpus. The near misses are what came back anyway."}
          </div>
          {result.has_evidence ? (
            <div className="flex flex-wrap gap-2" aria-label="Found in">
              {CHECKPOINTS.map((name) => (
                <span
                  key={name}
                  data-testid={`found-in-${name}`}
                  data-found={score.found_in?.[name] ? "yes" : "no"}
                >
                  <StatusChip
                    status={score.found_in?.[name] ? "succeeded" : "failed"}
                    label={`${CHECKPOINT_LABELS[name]}: ${score.found_in?.[name] ? "found" : "lost"}`}
                  />
                </span>
              ))}
            </div>
          ) : null}
        </div>
      </Section>
      <Suspense fallback={<Loading label="Loading the trace" />}>
        <QueryTrace evalRunId={evalRunId} itemId={itemId} />
      </Suspense>
      <Section
        title="Expected evidence"
        description="Gold labels name a document version, a section and a phrase."
      >
        {expected.length === 0 ? (
          <Empty>None.</Empty>
        ) : (
          <ul className="grid gap-2 text-sm">
            {expected.map((ref, index) => (
              <li key={index} className="grid gap-0.5">
                <span>
                  <code>
                    {String(ref.doc_key)}@{String(ref.version ?? "*")}
                  </code>{" "}
                  {String(ref.section ?? "")}{" "}
                  {score.findable?.[index] === false ? (
                    <StatusChip status="failed" label="not findable" />
                  ) : null}
                  {score.hidden?.[index] ? (
                    <StatusChip
                      status="flag"
                      label="hidden on the as-of date"
                    />
                  ) : null}
                </span>
                {ref.evidence ? (
                  <span className="whitespace-pre-wrap text-xs text-muted">
                    {String(ref.evidence)}
                  </span>
                ) : null}
              </li>
            ))}
          </ul>
        )}
      </Section>
      <Section
        title="Passages"
        description="Each checkpoint's list; evidence matched in a passage is marked."
        actions={
          <div className="flex flex-wrap gap-1" role="tablist">
            {CHECKPOINTS.map((name) => (
              <Button
                key={name}
                size="sm"
                variant={name === checkpoint ? "primary" : "secondary"}
                onPress={() => setCheckpoint(name)}
                aria-pressed={name === checkpoint}
              >
                {CHECKPOINT_LABELS[name]} ({lengths[name] ?? 0})
              </Button>
            ))}
          </div>
        }
      >
        {passages.length === 0 ? (
          <Empty>Nothing in this list.</Empty>
        ) : (
          <ol className="grid gap-3">
            {passages.map((passage) => (
              <li
                key={passage.chunk_id}
                className="grid gap-1"
                data-testid="eval-passage"
              >
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <span className="font-semibold">#{passage.rank}</span>
                  <code>
                    {passage.doc_key ?? passage.entry ?? "?"}@
                    {passage.version ?? "–"}
                  </code>
                  <span>{passage.section ?? ""}</span>
                  <span className="text-muted">
                    score {passage.score.toFixed(4)}
                  </span>
                  {RULES.map((name) =>
                    passage.relevant[name] ? (
                      <StatusChip key={name} status="succeeded" label={name} />
                    ) : null,
                  )}
                </div>
                <Highlighted passage={passage} />
              </li>
            ))}
          </ol>
        )}
      </Section>
    </div>
  );
}

type CompareOut = components["schemas"]["CompareOut"];
type MetricDelta = components["schemas"]["MetricDelta"];
type ChangeOut = components["schemas"]["ChangeOut"];

function formatP(value: number | null | undefined): string {
  if (value === null || value === undefined) return "–";
  return value < 0.001 ? "< 0.001" : value.toFixed(3);
}

function formatDelta(row: MetricDelta): string {
  if (row.delta === null || row.delta === undefined) return "–";
  const sign = row.delta > 0 ? "+" : "";
  return `${sign}${row.delta.toFixed(3)} (${(row.low ?? 0).toFixed(3)}–${(row.high ?? 0).toFixed(3)})`;
}

const WARNINGS: Record<string, string> = {
  few_queries:
    "Fewer than 50 queries pair up: intervals are wide, and small differences are noise.",
  different_splits:
    "The runs read different splits; only the items both scored are compared.",
};

function Diff({ label, changes }: { label: string; changes: ChangeOut[] }) {
  return changes.length === 0 ? (
    <Empty>{label}: the same.</Empty>
  ) : (
    <DataTable<ChangeOut>
      label={label}
      rows={changes}
      rowKey={(change) => change.path}
      columns={[
        { header: label, cell: (change) => <code>{change.path}</code> },
        { header: "Change", cell: (change) => change.op },
        { header: "A", cell: (change) => JSON.stringify(change.before) },
        { header: "B", cell: (change) => JSON.stringify(change.after) },
      ]}
    />
  );
}

/** Two eval runs, query by query. */
export function EvalCompare(props: { a: string | null; b: string | null }) {
  const { api } = useLab();
  const navigate = useNavigate();
  // The path holds a complete pair only; a half-made choice lives here until then.
  const [chosen, setChosen] = useState({ a: props.a, b: props.b });
  const { a, b } = chosen;
  const [rule, setRule] = useState<Rule>("knowledge-hit-v1");
  const [checkpoint, setCheckpoint] = useState<Checkpoint>("final");
  const finished = useLoad(
    async () =>
      unwrap(
        await api.GET("/eval-runs", {
          params: { query: { limit: 100, status: "succeeded" } },
        }),
      ),
    "eval-runs:succeeded",
  );
  const compared = useLoad<CompareOut | null>(
    async () =>
      a !== null && b !== null
        ? unwrap(
            await api.GET("/eval-comparisons", {
              params: { query: { a, b, rule, checkpoint } },
            }),
          )
        : null,
    `${a}:${b}:${rule}:${checkpoint}`,
  );
  const choose = (side: "a" | "b", value: string) => {
    const next = { ...chosen, [side]: value || null };
    setChosen(next);
    if (next.a !== null && next.b !== null) {
      navigate(
        { name: "eval-compare", a: next.a, b: next.b },
        { replace: true },
      );
    }
  };
  const options = finished.data?.items ?? [];
  const result = compared.data;
  const label = (run: EvalRunOut) =>
    `${shortHash(run.id, 8)} · ${refLabel(run.retrieval_ref)} · ${refLabel(run.eval_set)} (${run.split})`;
  const losses = (result?.queries ?? []).filter((row) => row.delta < 0);
  const wins = [...(result?.queries ?? [])]
    .filter((row) => row.delta > 0)
    .reverse();

  return (
    <div className="grid gap-4">
      <p className="text-sm">
        <Link to={{ name: "evals" }} className="text-link">
          Evals
        </Link>{" "}
        / Compare
      </p>
      <Section
        title="Compare two eval runs"
        description="B against A on the queries both scored: each delta with its 95% bootstrap interval and a paired randomization test."
      >
        <div className="flex flex-wrap items-end gap-2">
          {(["a", "b"] as const).map((side) => (
            <label key={side} className="grid gap-1 text-xs font-medium">
              {side === "a" ? "A (baseline)" : "B"}
              <select
                className={inputClass}
                aria-label={side === "a" ? "Eval run A" : "Eval run B"}
                value={(side === "a" ? a : b) ?? ""}
                onChange={(event) => choose(side, event.target.value)}
              >
                <option value="">Choose an eval run</option>
                {options.map((run) => (
                  <option key={run.id} value={run.id}>
                    {label(run)}
                  </option>
                ))}
              </select>
            </label>
          ))}
          <label className="grid gap-1 text-xs font-medium">
            Rule
            <select
              className={inputClass}
              value={rule}
              onChange={(event) => setRule(event.target.value as Rule)}
            >
              {RULES.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="grid gap-1 text-xs font-medium">
            Checkpoint
            <select
              className={inputClass}
              value={checkpoint}
              onChange={(event) =>
                setCheckpoint(event.target.value as Checkpoint)
              }
            >
              {CHECKPOINTS.map((item) => (
                <option key={item} value={item}>
                  {CHECKPOINT_LABELS[item]}
                </option>
              ))}
            </select>
          </label>
        </div>
        <ErrorNote error={compared.error} />
      </Section>
      {result === undefined && a !== null && b !== null ? <Loading /> : null}
      {result ? (
        <>
          <Section
            title="Deltas"
            description={`${result.paired} queries paired; ${result.resamples} resamples, seed ${result.seed}.`}
          >
            {result.warnings.map((warning) => (
              <p
                key={warning}
                role="note"
                className="mb-2 text-sm text-warning"
              >
                {WARNINGS[warning] ?? warning}
              </p>
            ))}
            <DataTable<MetricDelta>
              label="Deltas"
              rows={result.metrics}
              rowKey={(row) => row.metric}
              columns={[
                { header: "Metric", cell: (row) => row.metric },
                { header: "A", cell: (row) => row.mean_a?.toFixed(3) ?? "–" },
                { header: "B", cell: (row) => row.mean_b?.toFixed(3) ?? "–" },
                {
                  header: "B − A (95% CI)",
                  cell: (row) => (
                    <span data-testid="compare-delta">{formatDelta(row)}</span>
                  ),
                },
                { header: "p", cell: (row) => formatP(row.p_value) },
                { header: "p (Holm)", cell: (row) => formatP(row.p_holm) },
                {
                  header: "B wins / losses / ties",
                  cell: (row) =>
                    `${row.wins ?? 0} / ${row.losses ?? 0} / ${row.ties ?? 0}`,
                },
              ]}
            />
          </Section>
          <Section
            title="Queries"
            description={`Per query, by ${result.metrics[0]?.metric ?? "the first metric"}: where B lost, then where it won.`}
          >
            <div className="grid gap-4 md:grid-cols-2">
              {(
                [
                  ["B lost", losses],
                  ["B won", wins],
                ] as const
              ).map(([title, rows]) => (
                <div key={title} className="grid gap-1">
                  <span className="text-sm font-medium">
                    {title} ({rows.length})
                  </span>
                  {rows.length === 0 ? (
                    <Empty>None.</Empty>
                  ) : (
                    <ul className="grid gap-1 text-sm">
                      {rows.slice(0, 15).map((row) => (
                        <li key={row.item_id}>
                          {row.item_id}: {row.a.toFixed(2)} → {row.b.toFixed(2)}{" "}
                          (
                          <Link
                            to={{
                              name: "eval-query",
                              evalRunId: result.a.id,
                              itemId: row.item_id,
                            }}
                            className="text-link"
                          >
                            A
                          </Link>
                          ,{" "}
                          <Link
                            to={{
                              name: "eval-query",
                              evalRunId: result.b.id,
                              itemId: row.item_id,
                            }}
                            className="text-link"
                          >
                            B
                          </Link>
                          )
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              ))}
            </div>
          </Section>
          <Section
            title="Configuration"
            description="What B changed: its retrieval config and the pipeline behind its index."
          >
            <div className="grid gap-3">
              <Diff label="Retrieval config" changes={result.retrieval_diff} />
              <Diff label="Pipeline spec" changes={result.pipeline_diff} />
            </div>
          </Section>
        </>
      ) : null}
    </div>
  );
}
