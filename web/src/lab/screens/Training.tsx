import { Button } from "@heroui/react";
import { useEffect, useMemo, useState, type FormEvent } from "react";
import type { components } from "../api";
import { unwrap, useLoad } from "../api/load";
import { Chart } from "../components/Chart";
import { ParamForm } from "../components/ParamForm";
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
import { formatTime, shortHash } from "../format";
import { downloadText } from "../profiles";
import { defaultsFromSchema, type JsonSchema } from "../spec/schemaForm";
import {
  DEFAULT_SET_OPTIONS,
  EXPORT_FORMATS,
  MODEL_KINDS,
  cardSummary,
  epochsOf,
  epochsOption,
  isTrainingActive,
  metricText,
  negativeReasons,
  setCounts,
  variantName,
  withAdapter,
  withReranker,
} from "../training";

type TrainingSetOut = components["schemas"]["TrainingSetOut"];
type TrainingRunOut = components["schemas"]["TrainingRunOut"];
type TrainedModelOut = components["schemas"]["TrainedModelOut"];
type TrainingRecord = components["schemas"]["TrainingRecord"];

/** Reload every second while `active`. */
function usePolling(active: boolean, reload: () => void) {
  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(reload, 1000);
    return () => window.clearInterval(timer);
  }, [active, reload]);
}

function Counts({ rows }: { rows: { label: string; value: string }[] }) {
  return (
    <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-sm">
      {rows.map((row) => (
        <div key={row.label} className="contents">
          <dt className="text-muted">{row.label}</dt>
          <dd>{row.value}</dd>
        </div>
      ))}
    </dl>
  );
}

const LAB_ONLY =
  "Lab only: training and trained models need models.in_process, and a profile that uses one isn't deployable as it is.";

// --- the hub -------------------------------------------------------------------------

/** Training: mine training sets, train models on them, and the registry. */
export function Training() {
  const { api, host } = useLab();
  const navigate = useNavigate();
  const evals = useLoad(
    async () =>
      unwrap(
        await api.GET("/eval-runs", {
          params: { query: { status: "succeeded", limit: 100 } },
        }),
      ).items.filter((run) => run.split === "dev"),
    "dev-eval-runs",
  );
  const sets = useLoad(
    async () =>
      unwrap(
        await api.GET("/training-sets", { params: { query: { limit: 100 } } }),
      ).items,
    "training-sets",
  );
  const runs = useLoad(
    async () =>
      unwrap(
        await api.GET("/training-runs", { params: { query: { limit: 100 } } }),
      ).items,
    "training-runs",
  );
  const models = useLoad(
    async () =>
      unwrap(
        await api.GET("/trained-models", { params: { query: { limit: 100 } } }),
      ).items,
    "trained-models",
  );
  const active =
    (sets.data ?? []).some((set) => isTrainingActive(set.status)) ||
    (runs.data ?? []).some((run) => isTrainingActive(run.status));
  usePolling(active, () => {
    sets.reload();
    runs.reload();
    models.reload();
  });

  const [evalRunId, setEvalRunId] = useState("");
  const [name, setName] = useState("");
  const [maxNegatives, setMaxNegatives] = useState(8);
  const [validationShare, setValidationShare] = useState(0.2);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  const mine = async (event: FormEvent) => {
    event.preventDefault();
    const chosen = evalRunId || evals.data?.[0]?.id;
    if (!chosen) return;
    setBusy(true);
    setError(null);
    try {
      const created = unwrap(
        await api.POST("/training-sets", {
          body: {
            eval_run_id: chosen,
            ...(name ? { name } : {}),
            options: {
              ...DEFAULT_SET_OPTIONS,
              max_negatives: maxNegatives,
              validation_share: validationShare,
            },
          },
        }),
      );
      host.notify("Mining the training set.", "info");
      navigate({ name: "training-set", trainingSetId: created.id });
    } catch (failure) {
      setError(failure);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid gap-4">
      <Section
        title="Mine a training set"
        description="Positives from the gold evidence, hard negatives from what retrieval ranked high but isn't relevant. Only an eval run on the dev split: the test split stays locked for confirming a trained model once."
      >
        {evals.loading && !evals.data ? (
          <Loading />
        ) : (evals.data ?? []).length === 0 ? (
          <Empty>
            No succeeded dev eval runs yet. Run one on the{" "}
            <Link to={{ name: "evals" }} className="text-link">
              Evals
            </Link>{" "}
            screen first.
          </Empty>
        ) : (
          <form className="grid gap-2 sm:grid-cols-2" onSubmit={mine}>
            <label className="grid gap-1 text-sm">
              Eval run (dev)
              <select
                className={inputClass}
                value={evalRunId || evals.data?.[0]?.id || ""}
                onChange={(event) => setEvalRunId(event.target.value)}
              >
                {(evals.data ?? []).map((run) => (
                  <option key={run.id} value={run.id}>
                    {String(run.eval_set?.name ?? "eval set")}@
                    {String(run.eval_set?.version ?? "")} ·{" "}
                    {String(run.retrieval_ref?.name ?? "")} ·{" "}
                    {formatTime(run.created_at)}
                  </option>
                ))}
              </select>
            </label>
            <label className="grid gap-1 text-sm">
              Name (optional)
              <input
                className={inputClass}
                value={name}
                pattern="[a-z0-9][a-z0-9._\-]{0,63}"
                placeholder="the eval set's name"
                onChange={(event) => setName(event.target.value)}
              />
            </label>
            <label className="grid gap-1 text-sm">
              Hard negatives per item
              <input
                className={inputClass}
                type="number"
                min={0}
                max={200}
                value={maxNegatives}
                onChange={(event) =>
                  setMaxNegatives(Number(event.target.value))
                }
              />
            </label>
            <label className="grid gap-1 text-sm">
              Held out for validation
              <input
                className={inputClass}
                type="number"
                min={0}
                max={0.5}
                step={0.05}
                value={validationShare}
                onChange={(event) =>
                  setValidationShare(Number(event.target.value))
                }
              />
            </label>
            <div className="flex items-center gap-2 sm:col-span-2">
              <Button type="submit" isDisabled={busy} variant="primary">
                Mine
              </Button>
              <ErrorNote error={error} />
            </div>
          </form>
        )}
      </Section>

      <Section title="Training sets">
        <ErrorNote error={sets.error} />
        {sets.loading && !sets.data ? (
          <Loading />
        ) : (sets.data ?? []).length === 0 ? (
          <Empty>None yet.</Empty>
        ) : (
          <DataTable<TrainingSetOut>
            label="Training sets"
            rows={sets.data ?? []}
            rowKey={(row) => row.id}
            onSelect={(row) =>
              navigate({ name: "training-set", trainingSetId: row.id })
            }
            columns={[
              {
                header: "Name",
                cell: (row) => (
                  <Link
                    to={{ name: "training-set", trainingSetId: row.id }}
                    className="text-link"
                  >
                    {row.name}
                  </Link>
                ),
              },
              {
                header: "Status",
                cell: (row) => <StatusChip status={row.status} />,
              },
              {
                header: "Records",
                cell: (row) => String(row.stats?.records ?? "–"),
              },
              {
                header: "With evidence",
                cell: (row) => String(row.stats?.pairs ?? "–"),
              },
              {
                header: "Negatives",
                cell: (row) => String(row.stats?.negatives ?? "–"),
              },
              { header: "Created", cell: (row) => formatTime(row.created_at) },
            ]}
          />
        )}
      </Section>

      <Section title="Training runs">
        <ErrorNote error={runs.error} />
        {(runs.data ?? []).length === 0 ? (
          <Empty>None yet. Open a training set to train a model on it.</Empty>
        ) : (
          <DataTable<TrainingRunOut>
            label="Training runs"
            rows={runs.data ?? []}
            rowKey={(row) => row.id}
            onSelect={(row) =>
              navigate({ name: "training-run", trainingRunId: row.id })
            }
            columns={[
              { header: "Model", cell: (row) => row.name },
              { header: "Trainer", cell: (row) => row.trainer },
              {
                header: "Status",
                cell: (row) => <StatusChip status={row.status} />,
              },
              {
                header: "Epoch",
                cell: (row) =>
                  row.progress?.epoch !== undefined
                    ? `${String(row.progress.epoch)} of ${String(row.progress.epochs ?? "?")}`
                    : "–",
              },
              { header: "Created", cell: (row) => formatTime(row.created_at) },
            ]}
          />
        )}
      </Section>

      <Section title="Trained models" description={LAB_ONLY}>
        <ErrorNote error={models.error} />
        {(models.data ?? []).length === 0 ? (
          <Empty>None yet.</Empty>
        ) : (
          <DataTable<TrainedModelOut>
            label="Trained models"
            rows={models.data ?? []}
            rowKey={(row) => row.id}
            onSelect={(row) =>
              navigate({ name: "trained-model", modelId: row.id })
            }
            columns={[
              {
                header: "Model",
                cell: (row) => (
                  <Link
                    to={{ name: "trained-model", modelId: row.id }}
                    className="text-link"
                  >
                    {row.name}@{row.version}
                  </Link>
                ),
              },
              {
                header: "Kind",
                cell: (row) => MODEL_KINDS[row.kind] ?? row.kind,
              },
              {
                header: "Status",
                cell: (row) => (
                  <StatusChip status={row.stored ? row.status : "purged"} />
                ),
              },
              { header: "Base", cell: (row) => String(row.base.model ?? "–") },
              { header: "Created", cell: (row) => formatTime(row.created_at) },
            ]}
          />
        )}
      </Section>
    </div>
  );
}

// --- a training set --------------------------------------------------------------------

function RecordView({ record }: { record: TrainingRecord }) {
  return (
    <div className="grid gap-2">
      <PlainText text={record.query} label="Query" />
      <p className="text-xs text-muted">
        {record.split} · {(record.positives ?? []).length} positives ·{" "}
        {(record.negatives ?? []).length} negatives
        {record.abstention ? " · no evidence in the corpus" : ""}
      </p>
      {(record.positives ?? []).slice(0, 1).map((passage) => (
        <div key={passage.chunk_id} className="grid gap-1">
          <p className="text-xs font-medium">
            Positive: {passage.doc_key}@{passage.version} · {passage.section}
          </p>
          <PlainText text={passage.text} label="Positive passage" />
        </div>
      ))}
      {(record.negatives ?? []).slice(0, 1).map((passage) => (
        <div key={passage.chunk_id} className="grid gap-1">
          <p className="text-xs font-medium">
            Hardest negative ({passage.reason}, rank {passage.rank ?? "–"} in{" "}
            {passage.checkpoint ?? "–"}): {passage.doc_key}@{passage.version}
          </p>
          <PlainText text={passage.text} label="Negative passage" />
        </div>
      ))}
    </div>
  );
}

export function TrainingSetDetail({
  trainingSetId,
}: {
  trainingSetId: string;
}) {
  const { api, host } = useLab();
  const navigate = useNavigate();
  const found = useLoad(
    async () =>
      unwrap(
        await api.GET("/training-sets/{training_set_id}", {
          params: { path: { training_set_id: trainingSetId } },
        }),
      ),
    `training-set:${trainingSetId}`,
  );
  usePolling(
    found.data !== undefined && isTrainingActive(found.data.status),
    found.reload,
  );
  const succeeded = found.data?.status === "succeeded";
  const records = useLoad(
    async () =>
      succeeded
        ? unwrap(
            await api.GET("/training-sets/{training_set_id}/records", {
              params: {
                path: { training_set_id: trainingSetId },
                query: { limit: 20 },
              },
            }),
          )
        : null,
    `training-records:${trainingSetId}:${String(succeeded)}`,
  );
  const runs = useLoad(
    async () =>
      unwrap(
        await api.GET("/training-runs", {
          params: { query: { training_set_id: trainingSetId, limit: 50 } },
        }),
      ).items,
    `training-runs:${trainingSetId}`,
  );
  const trainers = useLoad(
    async () =>
      unwrap(
        await api.GET("/components", { params: { query: { stage: "train" } } }),
      ).items,
    "trainers",
  );
  const [shown, setShown] = useState(0);
  const [trainer, setTrainer] = useState("train/query-adapter");
  // Each trainer's edits over its schema's defaults (the form comes from the schema).
  const [edits, setEdits] = useState<Record<string, Record<string, unknown>>>(
    {},
  );
  const [modelName, setModelName] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const chosen = (trainers.data ?? []).find((item) => item.id === trainer);
  const schema = chosen?.params_schema as JsonSchema | undefined;
  const params = useMemo(
    () => ({
      ...(schema ? defaultsFromSchema(schema) : {}),
      ...(edits[trainer] ?? {}),
    }),
    [schema, edits, trainer],
  );

  if (found.loading && !found.data) return <Loading />;
  if (!found.data) return <ErrorNote error={found.error} />;
  const set = found.data;
  const reasons = negativeReasons(set.stats);

  const exportAs = async (format: string) => {
    try {
      const text = unwrap(
        await api.GET("/training-sets/{training_set_id}/export", {
          params: {
            path: { training_set_id: trainingSetId },
            query: { format: format as "sentence-transformers" },
          },
          parseAs: "text",
        }),
      ) as unknown as string;
      downloadText(text, `${set.name}-${format}.jsonl`, "application/x-ndjson");
    } catch (failure) {
      setError(failure);
    }
  };

  const train = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const started = unwrap(
        await api.POST("/training-runs", {
          body: {
            training_set_id: trainingSetId,
            name:
              modelName ||
              `${set.name}-${trainer.split("/")[1] ?? "model"}`.slice(0, 64),
            trainer: { component: trainer, params },
          },
        }),
      );
      host.notify("Training started.", "info");
      navigate({ name: "training-run", trainingRunId: started.id });
    } catch (failure) {
      setError(failure);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid gap-4">
      <Section
        title={`Training set ${set.name}`}
        description={
          <>
            Mined from{" "}
            <Link
              to={{ name: "eval", evalRunId: set.eval_run_id }}
              className="text-link"
            >
              a dev eval run
            </Link>
            {set.cached ? "; the records of an identical set were reused" : ""}.
          </>
        }
        actions={<StatusChip status={set.status} />}
      >
        {set.error_code ? (
          <p role="alert" className="text-sm text-danger">
            Failed: {set.error_code}
          </p>
        ) : null}
        {isTrainingActive(set.status) ? <Loading label="Mining" /> : null}
        <Counts rows={setCounts(set.stats)} />
        {reasons.length > 0 ? (
          <p className="mt-2 text-sm text-muted">
            Negatives by reason:{" "}
            {reasons
              .map((row) => `${row.reason.replace(/_/g, " ")} ${row.count}`)
              .join(", ")}
          </p>
        ) : null}
      </Section>

      {succeeded ? (
        <Section
          title="Exports"
          description="JSON Lines, each row naming its item and split."
          actions={EXPORT_FORMATS.map((item) => (
            <Button
              key={item.format}
              size="sm"
              variant="secondary"
              onPress={() => void exportAs(item.format)}
            >
              {item.label}
            </Button>
          ))}
        >
          <ul className="text-sm text-muted">
            {EXPORT_FORMATS.map((item) => (
              <li key={item.format}>
                {item.label}: {item.help}
              </li>
            ))}
          </ul>
        </Section>
      ) : null}

      {succeeded ? (
        <Section title="Train a model" description={LAB_ONLY}>
          <form className="grid gap-3" onSubmit={train}>
            <div className="grid gap-2 sm:grid-cols-2">
              <label className="grid gap-1 text-sm">
                Trainer
                <select
                  className={inputClass}
                  value={trainer}
                  onChange={(event) => setTrainer(event.target.value)}
                >
                  {(trainers.data ?? []).map((item) => (
                    <option
                      key={item.id}
                      value={item.id}
                      disabled={Boolean(item.unavailable_reason)}
                    >
                      {item.title}
                      {item.unavailable_reason ? " (not installed)" : ""}
                    </option>
                  ))}
                </select>
              </label>
              <label className="grid gap-1 text-sm">
                Model name
                <input
                  className={inputClass}
                  value={modelName}
                  pattern="[a-z0-9][a-z0-9._\-]{0,63}"
                  placeholder={`${set.name}-${trainer.split("/")[1] ?? "model"}`}
                  onChange={(event) => setModelName(event.target.value)}
                />
              </label>
            </div>
            {chosen ? (
              <p className="text-xs text-muted">
                {chosen.description} Runs on the {chosen.resource_class}{" "}
                workers.
              </p>
            ) : null}
            {schema ? (
              <ParamForm
                schema={schema}
                values={params}
                idPrefix={`train.${trainer}`}
                onChange={(field, value) =>
                  setEdits((current) => ({
                    ...current,
                    [trainer]: { ...(current[trainer] ?? {}), [field]: value },
                  }))
                }
              />
            ) : null}
            <div className="flex items-center gap-2">
              <Button type="submit" isDisabled={busy} variant="primary">
                Train
              </Button>
              <ErrorNote error={error} />
            </div>
          </form>
        </Section>
      ) : null}

      {(runs.data ?? []).length > 0 ? (
        <Section title="Training runs on this set">
          <DataTable<TrainingRunOut>
            label="Training runs on this set"
            rows={runs.data ?? []}
            rowKey={(row) => row.id}
            onSelect={(row) =>
              navigate({ name: "training-run", trainingRunId: row.id })
            }
            columns={[
              { header: "Model", cell: (row) => row.name },
              { header: "Trainer", cell: (row) => row.trainer },
              {
                header: "Status",
                cell: (row) => <StatusChip status={row.status} />,
              },
              { header: "Created", cell: (row) => formatTime(row.created_at) },
            ]}
          />
        </Section>
      ) : null}

      {records.data && records.data.items.length > 0 ? (
        <Section
          title="Records"
          description={`The first ${records.data.items.length} of ${records.data.total}. Text is shown as text.`}
          actions={
            <select
              aria-label="Record"
              className={inputClass}
              value={shown}
              onChange={(event) => setShown(Number(event.target.value))}
            >
              {records.data.items.map((record, index) => (
                <option key={record.item_id} value={index}>
                  {record.item_id}
                </option>
              ))}
            </select>
          }
        >
          {records.data.items[shown] ? (
            <RecordView record={records.data.items[shown]} />
          ) : null}
        </Section>
      ) : null}
    </div>
  );
}

// --- a training run ----------------------------------------------------------------------

export function TrainingRunDetail({
  trainingRunId,
}: {
  trainingRunId: string;
}) {
  const { api, host } = useLab();
  const found = useLoad(
    async () =>
      unwrap(
        await api.GET("/training-runs/{training_run_id}", {
          params: { path: { training_run_id: trainingRunId } },
        }),
      ),
    `training-run:${trainingRunId}`,
  );
  usePolling(
    found.data !== undefined && isTrainingActive(found.data.status),
    found.reload,
  );
  const [error, setError] = useState<unknown>(null);
  const metrics = found.data?.metrics ?? null;
  const epochs = useMemo(() => epochsOf(metrics), [metrics]);
  const option = useMemo(
    () =>
      epochsOption(
        epochs,
        (metrics?.best_epoch as number | null | undefined) ?? null,
        (metrics?.baseline as
          { "mrr@10"?: number | null } | null | undefined) ?? null,
      ),
    [epochs, metrics],
  );
  if (found.loading && !found.data) return <Loading />;
  if (!found.data) return <ErrorNote error={found.error} />;
  const run = found.data;
  const progress = run.progress ?? {};

  const cancel = async () => {
    const yes = await host.confirm({
      title: "Cancel this training run?",
      body: "It stops at its next step; nothing is registered.",
      confirmLabel: "Cancel the run",
      danger: true,
    });
    if (!yes) return;
    try {
      unwrap(
        await api.POST("/training-runs/{training_run_id}/cancel", {
          params: { path: { training_run_id: trainingRunId } },
        }),
      );
      found.reload();
    } catch (failure) {
      setError(failure);
    }
  };

  return (
    <div className="grid gap-4">
      <Section
        title={`Training run: ${run.name}`}
        description={
          <>
            {run.trainer} {run.trainer_version} on{" "}
            <Link
              to={{ name: "training-set", trainingSetId: run.training_set_id }}
              className="text-link"
            >
              its training set
            </Link>
            , from {String(run.base.model ?? "its base")}
            {run.base.revision ? ` (${String(run.base.revision)})` : ""}.
          </>
        }
        actions={
          <>
            <StatusChip status={run.status} />
            {isTrainingActive(run.status) ? (
              <Button size="sm" variant="danger" onPress={() => void cancel()}>
                Cancel
              </Button>
            ) : null}
          </>
        }
      >
        <ErrorNote error={error} />
        {run.error_code ? (
          <p role="alert" className="text-sm text-danger">
            Failed: {run.error_code}
          </p>
        ) : null}
        {isTrainingActive(run.status) ? (
          <p className="text-sm" data-testid="training-progress">
            {progress.epoch !== undefined
              ? `Epoch ${String(progress.epoch)} of ${String(progress.epochs ?? "?")}` +
                (progress.train_loss !== undefined
                  ? `: training loss ${metricText(progress.train_loss)}, validation MRR@10 ${metricText(progress["mrr@10"])}`
                  : "")
              : "Starting"}
          </p>
        ) : null}
        {run.model_id ? (
          <p className="text-sm">
            Registered as{" "}
            <Link
              to={{ name: "trained-model", modelId: run.model_id }}
              className="text-link"
            >
              {run.name}
            </Link>
            {run.model_cached ? " (an identical run's model, reused)" : ""}.
          </p>
        ) : null}
      </Section>
      {epochs.length > 0 ? (
        <Section
          title="Epochs"
          description="Training loss, and validation MRR@10 on the held-out items; the dashed line is the base model's, and the epoch kept is marked."
        >
          <Chart
            option={option}
            label="Training loss and validation MRR by epoch"
            testId="training-epochs"
            height={280}
          />
          <p className="mt-2 text-sm text-muted">
            Kept epoch {String(metrics?.best_epoch ?? "–")}
            {metrics?.stopped_early ? ", stopped early" : ""}.
          </p>
        </Section>
      ) : null}
      <Section title="Parameters">
        <PlainText
          text={JSON.stringify(run.params, null, 2)}
          label="Parameters"
        />
      </Section>
    </div>
  );
}

// --- a trained model -----------------------------------------------------------------------

export function TrainedModelDetail({ modelId }: { modelId: string }) {
  const { api, host } = useLab();
  const navigate = useNavigate();
  const found = useLoad(
    async () =>
      unwrap(
        await api.GET("/trained-models/{model_id}", {
          params: { path: { model_id: modelId } },
        }),
      ),
    `trained-model:${modelId}`,
  );
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  if (found.loading && !found.data) return <Loading />;
  if (!found.data) return <ErrorNote error={found.error} />;
  const model = found.data;
  const summary = cardSummary(model.card);
  const trainingSetId = String(
    (model.card.training_set as Record<string, unknown> | undefined)?.id ?? "",
  );
  const usable = model.status === "ready" && model.stored;

  const withdraw = async () => {
    const yes = await host.confirm({
      title: `Withdraw ${model.name}@${model.version}?`,
      body: "New runs can't use it; its record and card stay.",
      confirmLabel: "Withdraw",
      danger: true,
    });
    if (!yes) return;
    try {
      unwrap(
        await api.POST("/trained-models/{model_id}/withdraw", {
          params: { path: { model_id: modelId } },
        }),
      );
      found.reload();
    } catch (failure) {
      setError(failure);
    }
  };

  const download = async () => {
    try {
      const blob = unwrap(
        await api.GET("/trained-models/{model_id}/download", {
          params: { path: { model_id: modelId } },
          parseAs: "blob",
        }),
      ) as unknown as Blob;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${model.name}-v${model.version}.zip`;
      document.body.append(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (failure) {
      setError(failure);
    }
  };

  // The eval run the training set was mined from, with its retrieval config
  // changed to use this model: a dev eval run to compare with it.
  const evaluate = async () => {
    setBusy(true);
    setError(null);
    try {
      const set = unwrap(
        await api.GET("/training-sets/{training_set_id}", {
          params: { path: { training_set_id: trainingSetId } },
        }),
      );
      const base = unwrap(
        await api.GET("/eval-runs/{eval_run_id}", {
          params: { path: { eval_run_id: set.eval_run_id } },
        }),
      );
      const ref = base.retrieval_ref ?? {};
      const config = unwrap(
        await api.GET("/retrieval-configs/{name}/versions/{version}", {
          params: {
            path: { name: String(ref.name), version: Number(ref.version) },
          },
        }),
      );
      const name = variantName(config.name, model);
      const document =
        model.kind === "query-adapter"
          ? withAdapter(
              config.document as Record<string, unknown>,
              model.id,
              name,
            )
          : withReranker(
              config.document as Record<string, unknown>,
              model.id,
              name,
            );
      const saved = unwrap(
        await api.POST("/retrieval-configs", {
          body: document as components["schemas"]["RetrievalConfig"],
        }),
      );
      const started = unwrap(
        await api.POST("/eval-runs", {
          body: {
            run_id: base.run_id,
            retrieval_config_id: saved.id,
            eval_set_version_id: base.eval_set_version_id,
            split: "dev",
            confirm_test_read: false,
          },
        }),
      );
      host.notify(
        "Evaluating on the dev split; compare it with the base when it's done.",
        "info",
      );
      navigate({ name: "eval-compare", a: base.id, b: started.id });
    } catch (failure) {
      setError(failure);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid gap-4">
      <Section
        title={`${model.name}@${model.version}`}
        description={`${MODEL_KINDS[model.kind] ?? model.kind} from ${String(model.base.model ?? "its base")}. ${LAB_ONLY}`}
        actions={
          <>
            <StatusChip status={model.stored ? model.status : "purged"} />
            {model.kind !== "embedder" && usable && trainingSetId ? (
              <Button
                size="sm"
                variant="primary"
                isDisabled={busy}
                onPress={() => void evaluate()}
              >
                Evaluate on dev
              </Button>
            ) : null}
            {model.stored ? (
              <Button
                size="sm"
                variant="secondary"
                onPress={() => void download()}
              >
                Download
              </Button>
            ) : null}
            {model.status === "ready" ? (
              <Button
                size="sm"
                variant="danger"
                onPress={() => void withdraw()}
              >
                Withdraw
              </Button>
            ) : null}
          </>
        }
      >
        <ErrorNote error={error} />
        <Counts
          rows={[
            { label: "Use it", value: summary.how },
            { label: "Needs", value: summary.needs.join(", ") || "–" },
            {
              label: "Validation MRR@10",
              value: `${metricText(summary.baseline?.["mrr@10"])} base → ${metricText(summary.best?.["mrr@10"])} at epoch ${String(summary.bestEpoch ?? 0)}`,
            },
            {
              label: "Validation recall@10",
              value: `${metricText(summary.baseline?.["recall@10"])} → ${metricText(summary.best?.["recall@10"])}`,
            },
            { label: "Artifact", value: shortHash(model.model_key) },
            {
              label: "Withdrawn",
              value: model.withdrawn_at
                ? `${formatTime(model.withdrawn_at)} by ${model.withdrawn_by ?? "–"}`
                : "–",
            },
          ]}
        />
        <p className="mt-2 text-sm">
          <Link
            to={{ name: "training-run", trainingRunId: model.training_run_id }}
            className="text-link"
          >
            Its training run
          </Link>
          {trainingSetId ? (
            <>
              {" · "}
              <Link
                to={{ name: "training-set", trainingSetId }}
                className="text-link"
              >
                its training set
              </Link>
            </>
          ) : null}
        </p>
      </Section>
      <Section title="Limits">
        <ul className="list-disc pl-5 text-sm">
          {summary.limits.map((limit) => (
            <li key={limit}>{limit}</li>
          ))}
        </ul>
      </Section>
      <Section title="Model card" description="stratafi.model-card.v1">
        <PlainText
          text={JSON.stringify(model.card, null, 2)}
          label="Model card"
        />
      </Section>
    </div>
  );
}
