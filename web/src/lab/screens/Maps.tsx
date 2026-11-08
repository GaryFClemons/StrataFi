// Maps: every RAG pipeline the workspace has evaluated, and
// the map of any saved pipeline spec and retrieval config, with the datastores
// and services each step touches.

import { Button } from "@heroui/react";
import { useState, type FormEvent } from "react";
import { unwrap, useLoad } from "../api/load";
import { PipelineMap } from "../components/PipelineMap";
import {
  DataTable,
  Empty,
  ErrorNote,
  Loading,
  Section,
  inputClass,
} from "../components/ui";
import { Link, useLab, useNavigate } from "../context";
import { formatTime } from "../format";
import { formatValue } from "../tracking";

export function Maps() {
  const { api } = useLab();
  const navigate = useNavigate();
  const pipelines = useLoad(
    async () => unwrap(await api.GET("/rag-pipelines", {})),
    "rag-pipelines",
  );
  const specs = useLoad(
    async () =>
      unwrap(
        await api.GET("/pipeline-specs", { params: { query: { limit: 100 } } }),
      ),
    "pipeline-specs",
  );
  const configs = useLoad(
    async () =>
      unwrap(
        await api.GET("/retrieval-configs", {
          params: { query: { limit: 100 } },
        }),
      ),
    "retrieval-configs",
  );
  const [pipeline, setPipeline] = useState("");
  const [retrieval, setRetrieval] = useState("");

  const open = (event: FormEvent) => {
    event.preventDefault();
    navigate({
      name: "map",
      pipeline: pipeline || null,
      retrieval: retrieval || null,
    });
  };

  return (
    <div className="grid gap-4">
      <Section
        title="RAG pipelines"
        description="Every pipeline spec and retrieval config the workspace has evaluated together, with the services it calls and the datastores it reads and writes."
      >
        {pipelines.data === undefined ? (
          pipelines.error ? (
            <ErrorNote error={pipelines.error} />
          ) : (
            <Loading />
          )
        ) : pipelines.data.items.length === 0 ? (
          <Empty>
            No eval runs yet: evaluate a pipeline run with a retrieval config
            first.
          </Empty>
        ) : (
          <DataTable
            label="RAG pipelines"
            rows={pipelines.data.items}
            rowKey={(item) =>
              `${String(item.pipeline_spec.id)}:${String(item.retrieval_config.id)}`
            }
            columns={[
              {
                header: "Pipeline and config",
                cell: (item) => (
                  <Link
                    to={{
                      name: "map",
                      pipeline: String(item.pipeline_spec.id),
                      retrieval: String(item.retrieval_config.id),
                    }}
                    className="text-link"
                  >
                    {String(item.pipeline_spec.name)} v
                    {String(item.pipeline_spec.version)} ·{" "}
                    {String(item.retrieval_config.name)} v
                    {String(item.retrieval_config.version)}
                  </Link>
                ),
              },
              { header: "Eval runs", cell: (item) => item.eval_runs },
              {
                header: "Best recall",
                cell: (item) =>
                  item.best
                    ? `${formatValue(item.best.mean)} (${String(item.best.metric)}, ${String(item.best.eval_set)})`
                    : "–",
              },
              {
                header: "Services",
                cell: (item) => item.services.join(", ") || "none",
              },
              {
                header: "Datastores",
                cell: (item) => item.datastores.join(", "),
              },
              {
                header: "Text leaves the lab to",
                cell: (item) => item.sends_text_to.join(", ") || "nowhere",
              },
              { header: "Latest", cell: (item) => formatTime(item.latest_at) },
            ]}
          />
        )}
      </Section>
      <Section
        title="Map a pipeline"
        description="Any saved pipeline spec version, retrieval config version, or both."
      >
        <form
          onSubmit={open}
          aria-label="Map a pipeline"
          className="flex flex-wrap items-end gap-2"
        >
          <label className="grid gap-1 text-xs font-medium">
            Pipeline spec
            <select
              className={inputClass}
              value={pipeline}
              onChange={(event) => setPipeline(event.target.value)}
            >
              <option value="">None</option>
              {(specs.data?.items ?? []).map((spec) => (
                <option key={spec.id} value={spec.id}>
                  {spec.name} v{spec.version}
                </option>
              ))}
            </select>
          </label>
          <label className="grid gap-1 text-xs font-medium">
            Retrieval config
            <select
              className={inputClass}
              value={retrieval}
              onChange={(event) => setRetrieval(event.target.value)}
            >
              <option value="">None</option>
              {(configs.data?.items ?? []).map((config) => (
                <option key={config.id} value={config.id}>
                  {config.name} v{config.version}
                </option>
              ))}
            </select>
          </label>
          <Button type="submit" size="sm" isDisabled={!pipeline && !retrieval}>
            Map it
          </Button>
        </form>
      </Section>
    </div>
  );
}

export function MapView({
  pipeline,
  retrieval,
}: {
  pipeline: string | null;
  retrieval: string | null;
}) {
  const { api } = useLab();
  const found = useLoad(
    async () =>
      unwrap(
        await api.GET("/pipeline-maps", {
          params: {
            query: {
              pipeline_spec_id: pipeline ?? undefined,
              retrieval_config_id: retrieval ?? undefined,
            },
          },
        }),
      ),
    `${pipeline}:${retrieval}`,
  );
  const map = found.data;
  const title = map
    ? [map.pipeline, map.retrieval]
        .filter(Boolean)
        .map((ref) => `${String(ref?.name)} v${String(ref?.version)}`)
        .join(" · ")
    : "Map";
  return (
    <Section
      title={title}
      description="Select a step to see what it reads, writes and sends. Datastores are named, never addressed."
      actions={
        <Link to={{ name: "maps" }} className="text-link text-sm">
          Every RAG pipeline
        </Link>
      }
    >
      {map === undefined ? (
        found.error ? (
          <ErrorNote error={found.error} />
        ) : (
          <Loading />
        )
      ) : (
        <div className="grid gap-3">
          <p className="text-sm" data-testid="map-summary">
            {String(map.summary.stages)} ingest steps · datastores:{" "}
            {(map.summary.datastores as string[]).join(", ")} · services:{" "}
            {(map.summary.services as string[]).join(", ") || "none"}
            {(map.summary.sends_text_to as string[]).length
              ? ` · text leaves the lab to ${(map.summary.sends_text_to as string[]).join(", ")}`
              : ""}
          </p>
          <PipelineMap
            map={map}
            mermaid={map.mermaid}
            label={`The map of ${title}`}
          />
        </div>
      )}
    </Section>
  );
}

export default Maps;
