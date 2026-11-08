// Models: embedders, rerankers and adapters compared across
// an eval set version's eval runs, by their best runs and by controlled pairs
// (runs that differ only in the model); and the registries' aliases.

import { Button } from "@heroui/react";
import { useState, type FormEvent } from "react";
import type { components } from "../api";
import { unwrap, useLoad } from "../api/load";
import { Chart } from "../components/Chart";
import {
  DataTable,
  Empty,
  ErrorNote,
  Loading,
  Section,
  inputClass,
} from "../components/ui";
import { Link, useLab } from "../context";
import { modelBarsOption, pairMatrixOption } from "../models";
import { formatValue } from "../tracking";

type ModelComparisonOut = components["schemas"]["ModelComparisonOut"];
type Role = "embed" | "rerank" | "adapter";
const ROLES: { value: Role; label: string }[] = [
  { value: "embed", label: "Embedders" },
  { value: "rerank", label: "Rerankers" },
  { value: "adapter", label: "Query adapters" },
];

function Comparison({ data }: { data: ModelComparisonOut }) {
  const labels = new Map(data.models.map((row) => [row.key, row.label]));
  return (
    <div className="grid gap-4">
      {data.models.length ? (
        <Chart
          option={modelBarsOption(data.models, data.metric)}
          label={`Each model's best run's ${data.metric}, with its interval`}
          height={Math.max(160, 40 * data.models.length + 60)}
          testId="models-bars"
          renderer="svg"
        />
      ) : null}
      <DataTable
        label="Models"
        rows={data.models}
        rowKey={(row) => row.key}
        columns={[
          { header: "Model", cell: (row) => <code>{row.label}</code> },
          { header: "Component", cell: (row) => row.component },
          { header: "Runs", cell: (row) => row.runs },
          {
            header: `Best ${data.metric} (95% CI)`,
            cell: (row) =>
              row.best
                ? `${formatValue(row.best.mean)} (${formatValue(row.best.low)}–${formatValue(row.best.high)})`
                : "–",
          },
          {
            header: "Best run",
            cell: (row) =>
              row.best_eval_run_id ? (
                <Link
                  to={{ name: "eval", evalRunId: row.best_eval_run_id }}
                  className="text-link"
                >
                  open
                </Link>
              ) : (
                "–"
              ),
          },
          {
            header: "p95 ms (median)",
            cell: (row) => formatValue(row.latency_p95_ms, 1),
          },
          { header: "Egress", cell: (row) => row.egress ?? "–" },
        ]}
      />
      <Section
        title="Controlled pairs"
        description="Two eval runs that differ only in the model: their paired difference is the model's, with nothing else mixed in."
      >
        {data.pairs.length === 0 ? (
          <Empty>
            No two eval runs differ only in this role's model yet. Run the same
            pipeline and config with another model to measure it fairly.
          </Empty>
        ) : (
          <div className="grid gap-4">
            <Chart
              option={pairMatrixOption(
                data.models.map((row) => row.label),
                data.matrix,
                data.metric,
              )}
              label={`Controlled pairs: each column's model minus each row's, on ${data.metric}`}
              height={Math.max(200, 50 * data.models.length + 110)}
              testId="models-matrix"
              renderer="svg"
            />
            <DataTable
              label="Pairs"
              rows={data.pairs}
              rowKey={(pair) => `${pair.a_eval_run_id}:${pair.b_eval_run_id}`}
              columns={[
                {
                  header: "A",
                  cell: (pair) => labels.get(pair.a_model) ?? pair.a_model,
                },
                {
                  header: "B",
                  cell: (pair) => labels.get(pair.b_model) ?? pair.b_model,
                },
                {
                  header: "B minus A (95% CI)",
                  cell: (pair) =>
                    pair.result
                      ? `${formatValue(pair.result.delta)} (${formatValue(pair.result.low)}–${formatValue(pair.result.high)})`
                      : "–",
                },
                {
                  header: "p",
                  cell: (pair) => formatValue(pair.result?.p_value),
                },
                {
                  header: "Queries",
                  cell: (pair) => formatValue(pair.result?.n),
                },
                {
                  header: "Compare",
                  cell: (pair) => (
                    <Link
                      to={{
                        name: "compare",
                        runs: [pair.a_eval_run_id, pair.b_eval_run_id],
                      }}
                      className="text-link"
                    >
                      side by side
                    </Link>
                  ),
                },
              ]}
            />
          </div>
        )}
      </Section>
    </div>
  );
}

function Aliases() {
  const { api, host } = useLab();
  const aliases = useLoad(
    async () => unwrap(await api.GET("/registry-aliases", {})),
    "aliases",
  );
  const models = useLoad(
    async () =>
      unwrap(
        await api.GET("/trained-models", { params: { query: { limit: 100 } } }),
      ),
    "trained-models",
  );
  const [model, setModel] = useState("");
  const [alias, setAlias] = useState("champion");
  const [error, setError] = useState<unknown>(null);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    try {
      const set = unwrap(
        await api.PUT("/trained-models/{model_id}/aliases/{alias}", {
          params: { path: { model_id: model, alias } },
        }),
      );
      host.notify(
        `${set.name}@${set.alias} is version ${set.version}.`,
        "success",
      );
      aliases.reload();
      models.reload();
    } catch (failure) {
      setError(failure);
    }
  };

  return (
    <Section
      title="Aliases"
      description="A name such as champion for one version of a trained model or a knowledge profile; setting it again moves it. A profile's alias needs an owner."
    >
      <div className="grid gap-3">
        {aliases.data === undefined ? (
          <Loading />
        ) : aliases.data.items.length === 0 ? (
          <Empty>No aliases yet.</Empty>
        ) : (
          <DataTable
            label="Aliases"
            rows={aliases.data.items}
            rowKey={(item) => `${item.kind}:${item.name}:${item.alias}`}
            columns={[
              {
                header: "Alias",
                cell: (item) => <code>{`${item.name}@${item.alias}`}</code>,
              },
              {
                header: "Kind",
                cell: (item) =>
                  item.kind === "trained_model" ? "trained model" : "profile",
              },
              {
                header: "Version",
                cell: (item) =>
                  item.kind === "trained_model" ? (
                    <Link
                      to={{ name: "trained-model", modelId: item.target_id }}
                      className="text-link"
                    >
                      v{item.version ?? "?"}
                    </Link>
                  ) : (
                    <Link
                      to={{ name: "profile", profileId: item.target_id }}
                      className="text-link"
                    >
                      v{item.version ?? "?"}
                    </Link>
                  ),
              },
            ]}
          />
        )}
        <form
          onSubmit={submit}
          aria-label="Set an alias"
          className="flex flex-wrap items-end gap-2"
        >
          <label className="grid gap-1 text-xs font-medium">
            Trained model
            <select
              className={inputClass}
              value={model}
              onChange={(event) => setModel(event.target.value)}
            >
              <option value="">Choose a version</option>
              {(models.data?.items ?? []).map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name} v{item.version}
                  {item.aliases?.length ? ` (${item.aliases.join(", ")})` : ""}
                </option>
              ))}
            </select>
          </label>
          <label className="grid gap-1 text-xs font-medium">
            Alias
            <input
              className={inputClass}
              value={alias}
              pattern="[a-z][a-z0-9_\-]{0,31}"
              onChange={(event) => setAlias(event.target.value)}
            />
          </label>
          <Button type="submit" size="sm" isDisabled={!model || !alias}>
            Set alias
          </Button>
        </form>
        <ErrorNote error={error} />
      </div>
    </Section>
  );
}

export function Models() {
  const { api } = useLab();
  const sets = useLoad(
    async () =>
      unwrap(
        await api.GET("/eval-sets", { params: { query: { limit: 100 } } }),
      ),
    "eval-sets",
  );
  const frozen = (sets.data?.items ?? []).filter(
    (set) => set.latest_version?.frozen,
  );
  const [version, setVersion] = useState("");
  const [role, setRole] = useState<Role>("embed");
  const [metric, setMetric] = useState("recall@5");
  const chosen = version || frozen[0]?.latest_version?.id || "";
  const compared = useLoad(async () => {
    if (!chosen) return null;
    return unwrap(
      await api.GET("/model-comparisons", {
        params: { query: { eval_set_version_id: chosen, role, metric } },
      }),
    );
  }, `${chosen}:${role}:${metric}`);

  return (
    <div className="grid gap-4">
      <Section
        title="Compare models"
        description="Each model's best eval run on an eval set version, and its controlled pairs."
      >
        <div className="grid gap-4">
          <div className="flex flex-wrap items-end gap-2">
            <label className="grid gap-1 text-xs font-medium">
              Eval set
              <select
                className={inputClass}
                value={chosen}
                onChange={(event) => setVersion(event.target.value)}
              >
                {frozen.map((set) => (
                  <option key={set.id} value={set.latest_version?.id ?? ""}>
                    {set.name} v{set.latest_version?.version}
                  </option>
                ))}
              </select>
            </label>
            <label className="grid gap-1 text-xs font-medium">
              Role
              <select
                className={inputClass}
                value={role}
                onChange={(event) => setRole(event.target.value as Role)}
              >
                {ROLES.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="grid gap-1 text-xs font-medium">
              Metric
              <select
                className={inputClass}
                value={metric}
                onChange={(event) => setMetric(event.target.value)}
              >
                {["recall@5", "hit@5", "mrr@10", "ndcg@10", "recall@10"].map(
                  (name) => (
                    <option key={name}>{name}</option>
                  ),
                )}
              </select>
            </label>
          </div>
          {!chosen ? (
            <Empty>No frozen eval set yet.</Empty>
          ) : compared.data === undefined || compared.data === null ? (
            compared.error ? (
              <ErrorNote error={compared.error} />
            ) : (
              <Loading />
            )
          ) : compared.data.models.length === 0 ? (
            <Empty>
              No eval run of this eval set version used a model in this role.
            </Empty>
          ) : (
            <Comparison data={compared.data} />
          )}
        </div>
      </Section>
      <Aliases />
    </div>
  );
}

export default Models;
