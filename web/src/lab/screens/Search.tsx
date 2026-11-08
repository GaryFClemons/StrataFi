import { Button } from "@heroui/react";
import { useState, type FormEvent } from "react";
import type { components } from "../api";
import { unwrap, useLoad } from "../api/load";
import type { RunOut } from "../components/live";
import {
  Empty,
  ErrorNote,
  Loading,
  PlainText,
  Section,
  inputClass,
} from "../components/ui";
import { Link, useLab } from "../context";
import { ParamForm } from "../components/ParamForm";
import { highlightSegments } from "../evals";
import { formatTime, shortHash } from "../format";
import {
  DEFAULT_CONFIG,
  configOf,
  evidenceOf,
  retrievalDocument,
  type PlaygroundConfig,
} from "../playground";
import { reaches } from "../spec/editing";
import { defaultsFromSchema, type JsonSchema } from "../spec/schemaForm";

type SearchOut = components["schemas"]["SearchOut"];
type Hit = components["schemas"]["Hit"];
type ComponentOut = components["schemas"]["ComponentOut"];

const CHECKPOINT_LABELS: Record<string, string> = {
  dense: "Dense",
  keyword: "Keyword",
  fused: "Fused",
  candidates: "Candidates",
  reranked: "Reranked",
  final: "Final",
};
const CHECKPOINT_ORDER = [
  "final",
  "reranked",
  "candidates",
  "fused",
  "dense",
  "keyword",
];
type AskerIn = components["schemas"]["AskerIn"];

/** Who to search as: the Halcyon demo set's askers (fixtures/demo-corpus/retrieval.yaml),
 * whose groups are what document ACLs are checked against. An eval item brings its own. */
const ASKERS: AskerIn[] = [
  { name: "staff", roles: ["staff"], groups: ["all-staff"] },
  { name: "manager", roles: ["manager"], groups: ["all-staff", "managers"] },
  {
    name: "exec",
    roles: ["exec"],
    groups: ["all-staff", "managers", "executives"],
  },
];

function askerLabel(asker: AskerIn): string {
  const groups = asker.groups?.length ? asker.groups.join(", ") : "no groups";
  return `${asker.name ?? asker.user ?? "asker"} (${groups})`;
}

interface Shared {
  query: string;
  asker: AskerIn | null;
  asOf: string;
  evidence: string[];
}

function ConfigEditor({
  label,
  config,
  rerankers,
  onChange,
}: {
  label: string;
  config: PlaygroundConfig;
  rerankers: ComponentOut[];
  onChange: (config: PlaygroundConfig) => void;
}) {
  const { api } = useLab();
  const saved = useLoad(
    async () =>
      unwrap(
        await api.GET("/retrieval-configs", {
          params: { query: { limit: 50 } },
        }),
      ).items,
    "retrieval-configs",
  );
  const reranker = rerankers.find((item) => item.id === config.rerank);
  const set = (change: Partial<PlaygroundConfig>) =>
    onChange({ ...config, ...change });
  return (
    <fieldset
      className="grid gap-2 rounded-md border border-separator p-3 text-xs"
      aria-label={label}
    >
      <legend className="px-1 font-semibold">{label}</legend>
      {saved.data && saved.data.length ? (
        <label className="grid gap-1 font-medium">
          Start from a saved config
          <select
            className={inputClass}
            value=""
            onChange={(event) => {
              const found = saved.data?.find(
                (item) => item.id === event.target.value,
              );
              if (found)
                onChange(configOf(found.document as Record<string, unknown>));
            }}
          >
            <option value="">Choose…</option>
            {saved.data.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name} v{item.version}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      <div className="flex flex-wrap gap-3">
        <label className="flex items-center gap-1">
          <input
            type="checkbox"
            checked={config.dense}
            onChange={(event) => set({ dense: event.target.checked })}
          />
          Dense
        </label>
        <label className="flex items-center gap-1">
          <input
            type="checkbox"
            checked={config.keyword}
            onChange={(event) => set({ keyword: event.target.checked })}
          />
          Keyword
        </label>
        <label className="flex items-center gap-1">
          Fusion
          <select
            className={inputClass}
            value={config.fusion}
            onChange={(event) =>
              set({ fusion: event.target.value as PlaygroundConfig["fusion"] })
            }
          >
            <option value="rrf">RRF</option>
            <option value="weighted">Weighted</option>
          </select>
        </label>
        {config.fusion === "rrf" ? (
          <label className="flex items-center gap-1">
            k
            <input
              type="number"
              className={`${inputClass} w-16`}
              value={config.k}
              min={1}
              onChange={(event) => set({ k: Number(event.target.value) || 60 })}
            />
          </label>
        ) : (
          <label className="flex items-center gap-1">
            Dense weight
            <input
              type="number"
              className={`${inputClass} w-16`}
              value={config.alpha}
              min={0}
              max={1}
              step={0.1}
              onChange={(event) => set({ alpha: Number(event.target.value) })}
            />
          </label>
        )}
        <label className="flex items-center gap-1">
          Candidates
          <input
            type="number"
            className={`${inputClass} w-16`}
            value={config.candidates}
            min={1}
            onChange={(event) =>
              set({ candidates: Number(event.target.value) || 50 })
            }
          />
        </label>
        <label className="flex items-center gap-1">
          Results
          <input
            type="number"
            className={`${inputClass} w-16`}
            value={config.topK}
            min={1}
            max={50}
            onChange={(event) => set({ topK: Number(event.target.value) || 8 })}
          />
        </label>
      </div>
      <label className="grid gap-1 font-medium">
        Reranker
        <select
          className={inputClass}
          value={config.rerank}
          onChange={(event) => {
            const chosen = rerankers.find(
              (item) => item.id === event.target.value,
            );
            set({
              rerank: event.target.value,
              rerankParams: chosen
                ? defaultsFromSchema(chosen.params_schema as JsonSchema)
                : {},
            });
          }}
        >
          {rerankers.map((item) => (
            <option key={item.id} value={item.id}>
              {item.title}
              {item.available ? "" : " (not installed)"}
            </option>
          ))}
        </select>
      </label>
      {reranker ? (
        <ParamForm
          idPrefix={`${label}-rerank`}
          schema={reranker.params_schema as JsonSchema}
          values={config.rerankParams}
          onChange={(name, value) =>
            set({ rerankParams: { ...config.rerankParams, [name]: value } })
          }
        />
      ) : null}
    </fieldset>
  );
}

function HitText({ hit }: { hit: Hit }) {
  const spans = (hit.highlights ?? []) as [number, number][];
  if (!spans.length)
    return <PlainText text={hit.text} label={`Result ${hit.rank}`} />;
  return (
    <pre
      aria-label={`Result ${hit.rank}`}
      className="max-h-[32rem] overflow-auto whitespace-pre-wrap break-words rounded-md border border-separator p-3 font-mono text-xs"
    >
      {highlightSegments(hit.text, spans).map((segment, index) =>
        segment.marked ? (
          <mark key={index} data-testid="evidence" className="bg-warning-soft">
            {segment.text}
          </mark>
        ) : (
          <span key={index}>{segment.text}</span>
        ),
      )}
    </pre>
  );
}

function Results({ label, result }: { label: string; result: SearchOut }) {
  const names = [
    "final",
    ...CHECKPOINT_ORDER.filter(
      (name) => name !== "final" && name in result.checkpoints,
    ),
  ];
  const [checkpoint, setCheckpoint] = useState("final");
  const hits =
    checkpoint === "final"
      ? result.results
      : (result.checkpoints[checkpoint] ?? []);
  return (
    <section
      aria-label={`Results ${label}`}
      className="grid content-start gap-2"
    >
      <div
        role="tablist"
        aria-label={`Checkpoints ${label}`}
        className="flex flex-wrap gap-1"
      >
        {names.map((name) => (
          <button
            key={name}
            type="button"
            role="tab"
            aria-selected={checkpoint === name}
            className={`rounded-md px-3 py-1 text-sm ${
              checkpoint === name
                ? "bg-accent text-accent-foreground"
                : "bg-default"
            }`}
            onClick={() => setCheckpoint(name)}
          >
            {CHECKPOINT_LABELS[name] ?? name}
          </button>
        ))}
      </div>
      {hits.length === 0 ? (
        <Empty>Nothing matched.</Empty>
      ) : (
        <ol className="grid gap-2" aria-label={`Search results ${label}`}>
          {hits.map((hit) => (
            <li
              key={hit.chunk_id}
              className="grid gap-1"
              data-testid="search-hit"
            >
              <div className="flex flex-wrap gap-2 text-xs text-muted">
                <span>#{hit.rank}</span>
                <span data-testid="hit-entry">
                  {hit.entry ?? hit.document_id}
                </span>
                {hit.heading_path.length ? (
                  <span>{hit.heading_path.join(" › ")}</span>
                ) : null}
                {hit.page_start ? (
                  <span>
                    page {hit.page_start}
                    {hit.page_end && hit.page_end !== hit.page_start
                      ? `–${hit.page_end}`
                      : ""}
                  </span>
                ) : null}
                <span>score {hit.score.toFixed(4)}</span>
              </div>
              <HitText hit={hit} />
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

function EvalItemPicker({
  onPick,
}: {
  onPick: (shared: Partial<Shared>) => void;
}) {
  const { api } = useLab();
  const sets = useLoad(
    async () =>
      unwrap(await api.GET("/eval-sets", { params: { query: { limit: 50 } } }))
        .items,
    "playground-eval-sets",
  );
  const [setId, setSetId] = useState("");
  const chosen = sets.data?.find((item) => item.id === setId) ?? null;
  const items = useLoad(async () => {
    if (!chosen?.latest_version) return null;
    return unwrap(
      await api.GET("/eval-sets/{eval_set_id}/versions/{version}/items", {
        params: {
          path: {
            eval_set_id: chosen.id,
            version: chosen.latest_version.version,
          },
          query: { limit: 200 },
        },
      }),
    ).items;
  }, `playground-items:${setId}`);
  if (!sets.data || sets.data.length === 0) return null;
  return (
    <div className="flex flex-wrap items-end gap-2 text-xs">
      <label className="grid gap-1 font-medium">
        Eval set
        <select
          className={inputClass}
          value={setId}
          onChange={(event) => setSetId(event.target.value)}
        >
          <option value="">None</option>
          {sets.data.map((item) => (
            <option key={item.id} value={item.id}>
              {item.name}
            </option>
          ))}
        </select>
      </label>
      {items.data ? (
        <label className="grid gap-1 font-medium">
          Item
          <select
            className={inputClass}
            value=""
            onChange={(event) => {
              const item = items.data?.find(
                (found) => found.id === event.target.value,
              );
              if (!item) return;
              const picked = item.asker as AskerIn | string | null;
              const asker =
                picked && typeof picked === "object"
                  ? {
                      name: picked.name ?? null,
                      user: picked.user ?? null,
                      roles: picked.roles ?? [],
                      groups: picked.groups ?? [],
                    }
                  : null;
              const asOf = (
                chosen?.latest_version?.document as
                  { as_of?: string } | undefined
              )?.as_of;
              onPick({
                query: item.query,
                asker,
                asOf: asOf ?? "",
                evidence: evidenceOf(item.expected),
              });
            }}
          >
            <option value="">Choose an item…</option>
            {items.data.map((item) => (
              <option key={item.id} value={item.id}>
                {item.id}
              </option>
            ))}
          </select>
        </label>
      ) : null}
    </div>
  );
}

/** The search playground: a query with its asker and date, one or two retrieval configs,
 * and each one's checkpoints side by side. */
export function SearchPanel({ runId }: { runId: string }) {
  const { api } = useLab();
  const components = useLoad(
    async () =>
      unwrap(
        await api.GET("/components", {
          params: { query: { stage: "rerank" } },
        }),
      ).items,
    "rerank-components",
  );
  const [shared, setShared] = useState<Shared>({
    query: "",
    asker: null,
    asOf: "",
    evidence: [],
  });
  const [configs, setConfigs] = useState<PlaygroundConfig[]>([DEFAULT_CONFIG]);
  const [results, setResults] = useState<(SearchOut | null)[]>([]);
  const [errors, setErrors] = useState<unknown[]>([]);
  const [busy, setBusy] = useState(false);
  const labels = ["A", "B"];
  const choices =
    shared.asker &&
    !ASKERS.some((choice) => askerLabel(choice) === askerLabel(shared.asker!))
      ? [...ASKERS, shared.asker]
      : ASKERS;

  const search = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    const found = await Promise.all(
      configs.map(async (config, index) => {
        try {
          return {
            result: unwrap(
              await api.POST("/search", {
                body: {
                  run_id: runId,
                  query: shared.query,
                  mode: "hybrid",
                  top_k: config.topK,
                  retrieval_config: retrievalDocument(
                    config,
                    `playground-${labels[index]!.toLowerCase()}`,
                  ),
                  asker: shared.asker,
                  as_of: shared.asOf || null,
                  evidence: shared.evidence,
                },
              }),
            ),
            error: null,
          };
        } catch (failure) {
          return { result: null, error: failure };
        }
      }),
    );
    setResults(found.map((item) => item.result as unknown as SearchOut | null));
    setErrors(found.map((item) => item.error));
    setBusy(false);
  };

  return (
    <div className="grid gap-3">
      <EvalItemPicker
        onPick={(picked) => setShared((current) => ({ ...current, ...picked }))}
      />
      <form onSubmit={search} className="grid gap-3">
        <div className="flex flex-wrap items-end gap-2">
          <label className="grid flex-1 gap-1 text-xs font-medium">
            Query
            <input
              className={`${inputClass} min-w-64`}
              value={shared.query}
              onChange={(event) =>
                setShared({
                  ...shared,
                  query: event.target.value,
                  evidence: [],
                })
              }
              maxLength={2000}
              required
            />
          </label>
          <label className="grid gap-1 text-xs font-medium">
            Asker
            <select
              className={inputClass}
              value={shared.asker ? askerLabel(shared.asker) : ""}
              onChange={(event) =>
                setShared({
                  ...shared,
                  asker:
                    choices.find(
                      (choice) => askerLabel(choice) === event.target.value,
                    ) ?? null,
                })
              }
            >
              <option value="">Anyone (no ACL)</option>
              {choices.map((choice) => (
                <option key={askerLabel(choice)} value={askerLabel(choice)}>
                  {askerLabel(choice)}
                </option>
              ))}
            </select>
          </label>
          <label className="grid gap-1 text-xs font-medium">
            As of
            <input
              type="date"
              className={inputClass}
              value={shared.asOf}
              onChange={(event) =>
                setShared({ ...shared, asOf: event.target.value })
              }
            />
          </label>
          <label className="flex items-center gap-1 text-xs font-medium">
            <input
              type="checkbox"
              checked={configs.length === 2}
              onChange={(event) =>
                setConfigs(
                  event.target.checked
                    ? [configs[0]!, { ...configs[0]! }]
                    : [configs[0]!],
                )
              }
            />
            Compare two configs
          </label>
          <Button
            type="submit"
            size="sm"
            isDisabled={busy || shared.query.trim() === ""}
          >
            Search
          </Button>
        </div>
        {shared.evidence.length ? (
          <p className="text-xs text-muted">
            Evidence from the eval item is highlighted.
          </p>
        ) : null}
        <div
          className={`grid gap-3 ${configs.length === 2 ? "lg:grid-cols-2" : ""}`}
        >
          {configs.map((config, index) => (
            <ConfigEditor
              key={labels[index]}
              label={`Config ${labels[index]}`}
              config={config}
              rerankers={components.data ?? []}
              onChange={(next) =>
                setConfigs(
                  configs.map((item, position) =>
                    position === index ? next : item,
                  ),
                )
              }
            />
          ))}
        </div>
      </form>
      <div
        className={`grid gap-3 ${configs.length === 2 ? "lg:grid-cols-2" : ""}`}
      >
        {configs.map((_, index) => (
          <div key={labels[index]} className="grid content-start gap-2">
            <ErrorNote error={errors[index]} />
            {results[index] ? (
              <Results label={labels[index]!} result={results[index]!} />
            ) : null}
          </div>
        ))}
      </div>
    </div>
  );
}

/** Runs that built indexes: the ones search can use. */
export function useIndexedRuns() {
  const { api } = useLab();
  return useLoad(async () => {
    const page = unwrap(
      await api.GET("/runs", {
        params: { query: { status: "succeeded", limit: 50 } },
      }),
    );
    return page.items.filter((run) => reaches(run.until, "index"));
  }, "indexed-runs");
}

export function Search() {
  const runs = useIndexedRuns();
  const [chosen, setChosen] = useState<string | null>(null);
  const items: RunOut[] = runs.data ?? [];
  const runId = chosen ?? items[0]?.id ?? null;
  return (
    <Section
      title="Search"
      description="A playground over a run's indexes: retrievers, fusion and rerankers, two configurations side by side, every checkpoint."
    >
      <ErrorNote error={runs.error} />
      {runs.data === undefined ? (
        <Loading />
      ) : items.length === 0 ? (
        <Empty>
          No indexed runs yet: run a pipeline to Embed from{" "}
          <Link to={{ name: "pipelines", step: "embed" }} className="text-link">
            Pipelines
          </Link>
          .
        </Empty>
      ) : (
        <div className="grid gap-3">
          <label className="grid max-w-md gap-1 text-xs font-medium">
            Run
            <select
              className={inputClass}
              value={runId ?? ""}
              onChange={(event) => setChosen(event.target.value)}
            >
              {items.map((run) => (
                <option key={run.id} value={run.id}>
                  {shortHash(run.id, 8)} · {formatTime(run.created_at)}
                </option>
              ))}
            </select>
          </label>
          {runId ? <SearchPanel key={runId} runId={runId} /> : null}
        </div>
      )}
    </Section>
  );
}
