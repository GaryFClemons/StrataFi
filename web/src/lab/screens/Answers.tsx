import { Button } from "@heroui/react";
import { useEffect, useState, type FormEvent } from "react";
import type { components } from "../api";
import { unwrap, useLoad } from "../api/load";
import {
  agreementRows,
  automaticLabel,
  candidateCounts,
  formatKappa,
  isAnswerRunActive,
  labelsBy,
  metricRows,
  type AgreementRow,
  type Dimension,
  type MetricRow,
} from "../answers";
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
import { Link, useLab } from "../context";
import { formatEstimate } from "../evals";
import { formatTime, shortHash } from "../format";

type AnswerRunOut = components["schemas"]["AnswerRunOut"];
type AnswerResultOut = components["schemas"]["AnswerResultOut"];
type CandidateOut = components["schemas"]["CandidateOut"];
type CandidateStatus = CandidateOut["status"];

/** Reload every second while `active`. */
function usePolling(active: boolean, reload: () => void) {
  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(reload, 1000);
    return () => window.clearInterval(timer);
  }, [active, reload]);
}

/** The generate/ components, for a generator or judge select. */
function useGenerators() {
  const { api } = useLab();
  return useLoad(
    async () =>
      unwrap(
        await api.GET("/components", {
          params: { query: { stage: "generate" } },
        }),
      ).items,
    "generators",
  );
}

/** Drafted questions waiting for a person: the review queue. */
export function ReviewQueue({
  evalSetId,
  onAdded,
}: {
  evalSetId: string;
  onAdded: () => void;
}) {
  const { api, host } = useLab();
  const generators = useGenerators();
  const runs = useLoad(
    async () =>
      unwrap(
        await api.GET("/runs", {
          params: { query: { status: "succeeded", limit: 50 } },
        }),
      ).items,
    "succeeded-runs",
  );
  const [status, setStatus] = useState<CandidateStatus | "">("proposed");
  const candidates = useLoad(
    async () =>
      unwrap(
        await api.GET("/eval-sets/{eval_set_id}/candidates", {
          params: {
            path: { eval_set_id: evalSetId },
            query: status ? { status } : {},
          },
        }),
      ).items,
    `${evalSetId}:candidates:${status}`,
  );
  const everything = useLoad(
    async () =>
      unwrap(
        await api.GET("/eval-sets/{eval_set_id}/candidates", {
          params: { path: { eval_set_id: evalSetId } },
        }),
      ).items,
    `${evalSetId}:candidates`,
  );
  const [runId, setRunId] = useState("");
  const [generator, setGenerator] = useState("generate/fake");
  const [count, setCount] = useState(10);
  const [batchId, setBatchId] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [split, setSplit] = useState<"dev" | "test">("dev");
  const [error, setError] = useState<unknown>(null);

  const batch = useLoad(
    async () =>
      batchId === null
        ? null
        : unwrap(
            await api.GET(
              "/eval-sets/{eval_set_id}/question-batches/{batch_id}",
              {
                params: {
                  path: { eval_set_id: evalSetId, batch_id: batchId },
                },
              },
            ),
          ),
    `${evalSetId}:batch:${batchId ?? ""}`,
  );
  const drafting =
    batch.data !== undefined &&
    batch.data !== null &&
    isAnswerRunActive(batch.data.status);
  usePolling(drafting, batch.reload);
  const finished = batch.data?.status;
  useEffect(() => {
    if (finished === "succeeded" || finished === "failed") {
      candidates.reload();
      everything.reload();
    }
    // Reload once when the batch ends.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [finished]);

  const refresh = () => {
    candidates.reload();
    everything.reload();
  };

  const draft = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    try {
      const started = unwrap(
        await api.POST("/eval-sets/{eval_set_id}/question-batches", {
          params: { path: { eval_set_id: evalSetId } },
          body: {
            run_id: runId,
            generator: { component: generator },
            count,
            seed: 0,
          },
        }),
      );
      setBatchId(started.id);
    } catch (failure) {
      setError(failure);
    }
  };

  const decide = async (
    candidate: CandidateOut,
    decision: "accepted" | "rejected",
    edits: { query?: string; split?: "dev" | "test" } = {},
  ) => {
    setError(null);
    try {
      unwrap(
        await api.POST(
          "/eval-sets/{eval_set_id}/candidates/{candidate_id}/decision",
          {
            params: {
              path: { eval_set_id: evalSetId, candidate_id: candidate.id },
            },
            body: { decision, ...edits },
          },
        ),
      );
      setEditing(null);
      refresh();
    } catch (failure) {
      setError(failure);
    }
  };

  const add = async () => {
    setError(null);
    try {
      const added = unwrap(
        await api.POST("/eval-sets/{eval_set_id}/candidates/add", {
          params: { path: { eval_set_id: evalSetId } },
        }),
      );
      host.notify(
        `Added ${added.added.length} questions as v${added.version.version}.`,
        "success",
      );
      refresh();
      onAdded();
    } catch (failure) {
      setError(failure);
    }
  };

  const counts = candidateCounts(everything.data ?? []);
  return (
    <Section
      title="Review queue"
      description="Questions a generator drafted from a run's passages, each with the sentence that answers it. A person accepts (maybe edited) or rejects each one; only accepted ones join a new version."
      actions={
        <Button
          size="sm"
          isDisabled={!counts.accepted}
          onPress={() => void add()}
        >
          Add {counts.accepted ?? 0} accepted to a new version
        </Button>
      }
    >
      <form
        onSubmit={draft}
        aria-label="Draft questions"
        className="mb-3 flex flex-wrap items-end gap-2"
      >
        <label className="grid gap-1 text-xs font-medium">
          From run
          <select
            className={inputClass}
            value={runId}
            onChange={(event) => setRunId(event.target.value)}
            required
          >
            <option value="">choose a succeeded run</option>
            {(runs.data ?? []).map((run) => (
              <option key={run.id} value={run.id}>
                {shortHash(run.id, 8)} ({formatTime(run.created_at)})
              </option>
            ))}
          </select>
        </label>
        <label className="grid gap-1 text-xs font-medium">
          Generator
          <select
            className={inputClass}
            value={generator}
            onChange={(event) => setGenerator(event.target.value)}
          >
            {(generators.data ?? []).map((component) => (
              <option key={component.id} value={component.id}>
                {component.id}
              </option>
            ))}
          </select>
        </label>
        <label className="grid gap-1 text-xs font-medium">
          Questions
          <input
            type="number"
            min={1}
            max={100}
            className={inputClass}
            value={count}
            onChange={(event) => setCount(Number(event.target.value))}
          />
        </label>
        <Button type="submit" size="sm" variant="secondary">
          Draft questions
        </Button>
        {batch.data ? (
          <span className="text-xs" data-testid="batch-status">
            <StatusChip status={batch.data.status} />{" "}
            {batch.data.error_code ??
              `${String(batch.data.counts.drafted ?? 0)} drafted`}
          </span>
        ) : null}
      </form>
      <ErrorNote error={error ?? candidates.error} />
      <div className="mb-2 flex items-center gap-2 text-xs">
        <label className="flex items-center gap-1 font-medium">
          Show
          <select
            className={inputClass}
            value={status}
            onChange={(event) =>
              setStatus(event.target.value as CandidateStatus | "")
            }
          >
            <option value="proposed">proposed</option>
            <option value="accepted">accepted</option>
            <option value="rejected">rejected</option>
            <option value="added">added</option>
            <option value="">all</option>
          </select>
        </label>
        <span data-testid="queue-counts">
          {Object.entries(counts)
            .map(([name, value]) => `${value} ${name}`)
            .join(", ") || "empty"}
        </span>
      </div>
      {candidates.data === undefined ? (
        <Loading />
      ) : candidates.data.length === 0 ? (
        <Empty>Nothing here.</Empty>
      ) : (
        <DataTable<CandidateOut>
          label="Drafted questions"
          rows={candidates.data}
          rowKey={(candidate) => candidate.id}
          columns={[
            {
              header: "Question",
              cell: (candidate) =>
                editing === candidate.id ? (
                  <input
                    aria-label="Edited question"
                    className={`${inputClass} w-full`}
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                  />
                ) : (
                  <span>
                    {candidate.query}
                    {candidate.edited ? " (edited)" : ""}
                  </span>
                ),
            },
            {
              header: "Evidence",
              cell: (candidate) => (
                <blockquote className="text-xs italic">
                  {candidate.evidence}
                </blockquote>
              ),
            },
            {
              header: "Source",
              cell: (candidate) =>
                `${candidate.doc_key}@${candidate.version ?? "*"}${candidate.section ? `, ${candidate.section}` : ""}`,
            },
            {
              header: "Split",
              cell: (candidate) =>
                editing === candidate.id ? (
                  <select
                    aria-label="Split"
                    className={inputClass}
                    value={split}
                    onChange={(event) =>
                      setSplit(event.target.value as "dev" | "test")
                    }
                  >
                    <option value="dev">dev</option>
                    <option value="test">test</option>
                  </select>
                ) : (
                  candidate.split
                ),
            },
            {
              header: "Status",
              cell: (candidate) => <StatusChip status={candidate.status} />,
            },
            {
              header: "",
              cell: (candidate) =>
                candidate.status === "added" ? (
                  <code>{candidate.item_id}</code>
                ) : editing === candidate.id ? (
                  <span className="flex gap-1">
                    <Button
                      size="sm"
                      onPress={() =>
                        void decide(candidate, "accepted", { query, split })
                      }
                    >
                      Save and accept
                    </Button>
                    <Button
                      size="sm"
                      variant="secondary"
                      onPress={() => setEditing(null)}
                    >
                      Cancel
                    </Button>
                  </span>
                ) : (
                  <span className="flex gap-1">
                    <Button
                      size="sm"
                      onPress={() => void decide(candidate, "accepted")}
                    >
                      Accept
                    </Button>
                    <Button
                      size="sm"
                      variant="secondary"
                      onPress={() => {
                        setEditing(candidate.id);
                        setQuery(candidate.query);
                        setSplit(candidate.split);
                      }}
                    >
                      Edit
                    </Button>
                    <Button
                      size="sm"
                      variant="danger"
                      onPress={() => void decide(candidate, "rejected")}
                    >
                      Reject
                    </Button>
                  </span>
                ),
            },
          ]}
        />
      )}
    </Section>
  );
}

function metricOf(run: AnswerRunOut, key: string): string {
  const row = metricRows(run.metrics).find((metric) => metric.key === key);
  return formatEstimate(row?.estimate ?? null);
}

/** An eval run's answer runs, and a new one. */
export function AnswerRuns({ evalRunId }: { evalRunId: string }) {
  const { api } = useLab();
  const generators = useGenerators();
  const runs = useLoad(
    async () =>
      unwrap(
        await api.GET("/answer-runs", {
          params: { query: { eval_run_id: evalRunId } },
        }),
      ).items,
    `${evalRunId}:answers`,
  );
  const [generator, setGenerator] = useState("generate/fake");
  const [judge, setJudge] = useState("generate/fake");
  const [error, setError] = useState<unknown>(null);
  usePolling(
    (runs.data ?? []).some((run) => isAnswerRunActive(run.status)),
    runs.reload,
  );

  const start = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    try {
      unwrap(
        await api.POST("/answer-runs", {
          body: {
            eval_run_id: evalRunId,
            generator: { component: generator },
            judge: judge ? { component: judge } : null,
            passages: 8,
          },
        }),
      );
      runs.reload();
    } catch (failure) {
      setError(failure);
    }
  };

  return (
    <Section
      title="Answers"
      description="A generator answers each query from its final passages with the pinned grounded-qa prompt; a judge scores faithfulness and correctness. generate/gateway answers through the model gateway's chat models; generate/fake is the lab's stand-in."
    >
      <form
        onSubmit={start}
        aria-label="Answer the queries"
        className="mb-3 flex flex-wrap items-end gap-2"
      >
        <label className="grid gap-1 text-xs font-medium">
          Generator
          <select
            className={inputClass}
            value={generator}
            onChange={(event) => setGenerator(event.target.value)}
          >
            {(generators.data ?? []).map((component) => (
              <option key={component.id} value={component.id}>
                {component.id}
              </option>
            ))}
          </select>
        </label>
        <label className="grid gap-1 text-xs font-medium">
          Judge
          <select
            className={inputClass}
            value={judge}
            onChange={(event) => setJudge(event.target.value)}
          >
            <option value="">none (graders only)</option>
            {(generators.data ?? []).map((component) => (
              <option key={component.id} value={component.id}>
                {component.id}
              </option>
            ))}
          </select>
        </label>
        <Button type="submit" size="sm" variant="secondary">
          Answer the queries
        </Button>
      </form>
      <ErrorNote error={error ?? runs.error} />
      {runs.data === undefined ? (
        <Loading />
      ) : runs.data.length === 0 ? (
        <Empty>No answers yet.</Empty>
      ) : (
        <DataTable<AnswerRunOut>
          label="Answer runs"
          rows={runs.data}
          rowKey={(run) => run.id}
          columns={[
            {
              header: "Run",
              cell: (run) => (
                <Link
                  to={{ name: "answers", answerRunId: run.id }}
                  className="text-link"
                >
                  <code>{shortHash(run.id, 8)}</code>
                </Link>
              ),
            },
            {
              header: "Status",
              cell: (run) => (
                <StatusChip
                  status={run.status}
                  label={
                    run.error_code
                      ? `${run.status} (${run.error_code.replace(/_/g, " ")})`
                      : undefined
                  }
                />
              ),
            },
            {
              header: "Generator",
              cell: (run) =>
                String(
                  (run.config.generator as Record<string, unknown>).component,
                ),
            },
            {
              header: "Abstention accuracy",
              cell: (run) => metricOf(run, "abstention_accuracy"),
            },
            {
              header: "Citation precision",
              cell: (run) => metricOf(run, "citation_precision"),
            },
            {
              header: "Faithful",
              cell: (run) => metricOf(run, "faithfulness_pass"),
            },
            { header: "Started", cell: (run) => formatTime(run.created_at) },
          ]}
        />
      )}
    </Section>
  );
}

const DIMENSION_TEXT: Record<Dimension, string> = {
  faithful: "Faithful",
  correct: "Correct",
  abstained: "Abstained",
};

/** One answer run: metrics, people's labels against the judge, each answer. */
export function AnswerRunDetail({ answerRunId }: { answerRunId: string }) {
  const { api } = useLab();
  const path = { params: { path: { answer_run_id: answerRunId } } };
  const me = useLoad(async () => unwrap(await api.GET("/me")), "me");
  const run = useLoad(
    async () => unwrap(await api.GET("/answer-runs/{answer_run_id}", path)),
    answerRunId,
  );
  const active = run.data !== undefined && isAnswerRunActive(run.data.status);
  usePolling(active, run.reload);
  const done = run.data?.status === "succeeded";
  const results = useLoad(
    async () =>
      done
        ? unwrap(await api.GET("/answer-runs/{answer_run_id}/queries", path))
            .items
        : [],
    `${answerRunId}:queries:${String(done)}`,
  );
  const agreement = useLoad(
    async () =>
      done
        ? unwrap(await api.GET("/answer-runs/{answer_run_id}/agreement", path))
            .dimensions
        : {},
    `${answerRunId}:agreement:${String(done)}`,
  );
  const [shown, setShown] = useState<string | null>(null);
  const detail = useLoad(
    async () =>
      shown === null
        ? null
        : unwrap(
            await api.GET("/answer-runs/{answer_run_id}/queries/{item_id}", {
              params: { path: { answer_run_id: answerRunId, item_id: shown } },
            }),
          ),
    `${answerRunId}:detail:${shown ?? ""}`,
  );
  const [error, setError] = useState<unknown>(null);

  const label = async (
    itemId: string,
    dimension: Dimension,
    value: boolean,
  ) => {
    setError(null);
    try {
      unwrap(
        await api.PUT("/answer-runs/{answer_run_id}/labels", {
          ...path,
          body: { item_id: itemId, dimension, label: value },
        }),
      );
      results.reload();
      agreement.reload();
    } catch (failure) {
      setError(failure);
    }
  };

  if (run.data === undefined) {
    return run.error ? <ErrorNote error={run.error} /> : <Loading />;
  }
  const subject = me.data?.subject;
  return (
    <div className="grid gap-4">
      <p className="text-sm">
        <Link
          to={{ name: "eval", evalRunId: run.data.eval_run_id }}
          className="text-link"
        >
          Eval run {shortHash(run.data.eval_run_id, 8)}
        </Link>{" "}
        / answers {shortHash(answerRunId, 8)}
      </p>
      <Section
        title={`Answers ${shortHash(answerRunId, 8)}`}
        description="Means over queries with 95% intervals. Prompts are pinned by version and hash."
        actions={<StatusChip status={run.data.status} />}
      >
        {run.data.error_code ? (
          <p className="text-sm text-danger">
            Failed: {run.data.error_code.replace(/_/g, " ")}
          </p>
        ) : null}
        <DataTable<MetricRow>
          label="Answer metrics"
          rows={metricRows(run.data.metrics)}
          rowKey={(row) => row.key}
          columns={[
            { header: "Metric", cell: (row) => row.label },
            {
              header: "Mean (95% CI)",
              cell: (row) => formatEstimate(row.estimate),
            },
            { header: "Queries", cell: (row) => row.estimate?.n ?? "–" },
            { header: "Means", cell: (row) => row.help },
          ]}
        />
      </Section>
      <Section
        title="Calibration"
        description="How often the judge (faithful, correct) and the abstention detector agree with people's labels, with Cohen's kappa."
      >
        <DataTable<AgreementRow>
          label="Agreement with labels"
          rows={agreementRows(
            (agreement.data ?? {}) as Record<string, Record<string, unknown>>,
          )}
          rowKey={(row) => row.dimension}
          columns={[
            {
              header: "Dimension",
              cell: (row) => DIMENSION_TEXT[row.dimension as Dimension],
            },
            { header: "Labels", cell: (row) => row.n },
            {
              header: "Agreement (95% CI)",
              cell: (row) => formatEstimate(row.agreement),
            },
            { header: "Kappa", cell: (row) => formatKappa(row.kappa) },
          ]}
        />
      </Section>
      <Section
        title="Each answer"
        description="Label an answer to add it to the calibration set; open it to read it with its passages."
      >
        <ErrorNote error={error ?? results.error} />
        {results.data === undefined ? (
          <Loading />
        ) : results.data.length === 0 ? (
          <Empty>No answers yet.</Empty>
        ) : (
          <DataTable<AnswerResultOut>
            label="Answers"
            rows={results.data}
            rowKey={(row) => row.item_id}
            selected={shown}
            onSelect={(row) => setShown(row.item_id)}
            columns={[
              { header: "Item", cell: (row) => <code>{row.item_id}</code> },
              {
                header: "Abstained",
                cell: (row) => (row.score.abstained ? "yes" : "no"),
              },
              {
                header: "Citations",
                cell: (row) => String(row.score.citations ?? 0),
              },
              {
                header: "Faithfulness",
                cell: (row) => String(row.score.faithfulness ?? "–"),
              },
              ...(["faithful", "abstained"] as Dimension[]).map(
                (dimension) => ({
                  header: `${DIMENSION_TEXT[dimension]}: automatic, yours`,
                  cell: (row: AnswerResultOut) => {
                    const mine = labelsBy(row.labels, subject)[dimension];
                    const automatic = automaticLabel(row.score, dimension);
                    return (
                      <span className="flex items-center gap-1">
                        <span className="text-xs">
                          {automatic === null ? "–" : automatic ? "yes" : "no"}
                        </span>
                        <Button
                          size="sm"
                          variant={mine === true ? "primary" : "secondary"}
                          aria-label={`${DIMENSION_TEXT[dimension]}: yes for ${row.item_id}`}
                          onPress={() =>
                            void label(row.item_id, dimension, true)
                          }
                        >
                          Yes
                        </Button>
                        <Button
                          size="sm"
                          variant={mine === false ? "primary" : "secondary"}
                          aria-label={`${DIMENSION_TEXT[dimension]}: no for ${row.item_id}`}
                          onPress={() =>
                            void label(row.item_id, dimension, false)
                          }
                        >
                          No
                        </Button>
                      </span>
                    );
                  },
                }),
              ),
            ]}
          />
        )}
      </Section>
      {detail.data ? (
        <Section title={`Answer to ${detail.data.item_id}`}>
          <PlainText text={detail.data.answer} label="Answer" />
          <DataTable
            label="Passages given"
            rows={detail.data.passages}
            rowKey={(passage) => String(passage.source_id)}
            columns={[
              {
                header: "Source",
                cell: (passage) => <code>{String(passage.source_id)}</code>,
              },
              {
                header: "Relevant",
                cell: (passage) => (passage.relevant ? "yes" : "no"),
              },
              {
                header: "Text",
                cell: (passage) => (
                  <span className="text-xs">{String(passage.text)}</span>
                ),
              },
            ]}
          />
        </Section>
      ) : null}
    </div>
  );
}
