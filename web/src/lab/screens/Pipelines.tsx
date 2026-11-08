import { Button } from "@heroui/react";
import { useState } from "react";
import type { ApiClient, components } from "../api";
import { ApiError, unwrap, useLoad } from "../api/load";
import { useLiveRun } from "../components/live";
import {
  Empty,
  ErrorNote,
  Loading,
  Section,
  inputClass,
} from "../components/ui";
import { Link, useLab } from "../context";
import { shortHash } from "../format";
import type { StepId } from "../routes";
import {
  SAMPLE_SIZES,
  STEPS,
  reaches,
  sampleFor,
  slotsFor,
  stepOf,
  type SpecDocument,
} from "../spec/editing";
import { EmbedInspector, ModelCatalog } from "./steps/Embed";
import { ChunkInspector } from "./steps/Chunk";
import {
  ParseInspector,
  ParseRoutes,
  type Workbench as ParseWorkbench,
} from "./steps/Parse";
import { PolicyEditor, RuleList, ScanInspector } from "./steps/ScanClean";
import { ExtractInspector, ExtractSources } from "./steps/Extract";
import { RunProgress } from "./Runs";
import { SearchPanel } from "./Search";
import { SlotEditor } from "../components/SlotEditor";

type PipelineSpec = components["schemas"]["PipelineSpec"];

interface StepRun {
  step: StepId;
  runId: string;
  until: string | null;
}

interface Workbench {
  specName: string;
  document: SpecDocument | null;
  corpusVersionId: string | null;
  sampleSize: number | null;
  /** The runs started from this screen, newest first. */
  runs: StepRun[];
}

const STORAGE_KEY = "stratafi.workbench";
const EMPTY: Workbench = {
  specName: "lab",
  document: null,
  corpusVersionId: null,
  sampleSize: 5,
  runs: [],
};

function restore(): Workbench {
  try {
    const stored = window.sessionStorage.getItem(STORAGE_KEY);
    return stored
      ? { ...EMPTY, ...(JSON.parse(stored) as Partial<Workbench>) }
      : EMPTY;
  } catch {
    return EMPTY;
  }
}

function keep(workbench: Workbench): void {
  try {
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(workbench));
  } catch {
    // Not kept: the workbench lasts until the tab reloads.
  }
}

/** The latest saved version of a spec name, or the baseline when there is none (or asked). */
async function fetchSpec(
  api: ApiClient,
  name: string,
  baseline: boolean,
): Promise<SpecDocument | null> {
  if (!baseline) {
    const latest = await api.GET("/pipeline-specs/{name}", {
      params: { path: { name } },
    });
    if (latest.data) return latest.data.document as SpecDocument;
    if (latest.response.status !== 404) unwrap(latest);
  }
  // An empty spec resolves to the baseline: every default filled in.
  const resolved = unwrap(
    await api.POST("/pipeline-specs/validate", {
      body: { schema: "stratafi.pipeline.v1", name } as PipelineSpec,
    }),
  );
  return (resolved.document ?? null) as SpecDocument | null;
}

/** The run a step shows: the newest that went at least as far as the step. */
function runFor(step: StepId, runs: StepRun[]): StepRun | null {
  const last = stepOf(step).until ?? "index";
  return runs.find((run) => reaches(run.until, last)) ?? null;
}

function useComponents() {
  const { api } = useLab();
  return useLoad(
    async () => unwrap(await api.GET("/components")).items,
    "components",
  );
}

/** Every frozen corpus version, newest first within each corpus. */
function useCorpusVersions() {
  const { api } = useLab();
  return useLoad(async () => {
    const corpora = unwrap(
      await api.GET("/corpora", { params: { query: { limit: 100 } } }),
    );
    const options: { id: string; label: string }[] = [];
    for (const corpus of corpora.items) {
      const versions = unwrap(
        await api.GET("/corpora/{corpus_id}/versions", {
          params: { path: { corpus_id: corpus.id }, query: { limit: 20 } },
        }),
      );
      for (const version of versions.items) {
        options.push({
          id: version.id,
          label: `${corpus.name} v${version.version} (${version.document_count} documents)`,
        });
      }
    }
    return options;
  }, "corpus-versions");
}

function Inspector({
  step,
  runId,
  workbench,
}: {
  step: StepId;
  runId: string;
  workbench: ParseWorkbench;
}) {
  const live = useLiveRun(runId);
  const run = live.data;
  if (run === undefined)
    return live.error ? <ErrorNote error={live.error} /> : <Loading />;
  const inspector =
    step === "extract" ? (
      <ExtractInspector run={run} />
    ) : step === "scan" ? (
      <ScanInspector run={run} />
    ) : step === "parse" ? (
      <ParseInspector run={run} workbench={workbench} />
    ) : step === "chunk" ? (
      <ChunkInspector run={run} workbench={workbench} />
    ) : step === "embed" ? (
      <EmbedInspector run={run} />
    ) : null;
  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <RunProgress run={run} />
        <Link to={{ name: "run", runId: run.id }} className="text-sm text-link">
          Run <code>{shortHash(run.id, 8)}</code>
        </Link>
      </div>
      {inspector}
    </div>
  );
}

export function Pipelines({ step }: { step: StepId }) {
  const { api, host } = useLab();
  const [workbench, setWorkbench] = useState<Workbench>(restore);
  const [nameDraft, setNameDraft] = useState(workbench.specName);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const components = useComponents();
  const versions = useCorpusVersions();
  const definition = stepOf(step);

  const update = (change: Partial<Workbench>) =>
    setWorkbench((current) => {
      const next = { ...current, ...change };
      keep(next);
      return next;
    });

  /** The latest saved version of a spec name, or the baseline when there is none. */
  const load = async (name: string, baseline = false) => {
    setError(null);
    try {
      update({
        specName: name,
        document: await fetchSpec(api, name, baseline),
      });
    } catch (failure) {
      setError(failure);
    }
  };

  // The first visit starts from the saved spec, or the baseline.
  const initial = useLoad(
    async () => workbench.document ?? fetchSpec(api, workbench.specName, false),
    "initial-spec",
  );

  const corpusVersionId =
    workbench.corpusVersionId ?? versions.data?.[0]?.id ?? null;

  const runToStep = async () => {
    const document = workbench.document ?? initial.data;
    if (!document || !corpusVersionId || definition.until === null) return;
    setBusy(true);
    setError(null);
    try {
      const saved = unwrap(
        await api.POST("/pipeline-specs", {
          body: { ...document, name: workbench.specName } as PipelineSpec,
        }),
      );
      const run = unwrap(
        await api.POST("/runs", {
          body: {
            pipeline_spec_id: saved.id,
            corpus_version_id: corpusVersionId,
            until: definition.until,
            sample: sampleFor(workbench.sampleSize),
            mode: "interactive",
          },
        }),
      );
      update({
        document: saved.document as SpecDocument,
        corpusVersionId,
        runs: [
          { step, runId: run.id, until: run.until ?? null },
          ...workbench.runs,
        ].slice(0, 20),
      });
      host.notify(
        `Running to ${definition.title} (spec v${saved.version}).`,
        "info",
      );
    } catch (failure) {
      setError(failure);
      if (failure instanceof ApiError && failure.code === "invalid_spec") {
        host.notify(
          "The pipeline spec isn't valid; check the highlighted settings.",
          "danger",
        );
      }
    } finally {
      setBusy(false);
    }
  };

  const shown = runFor(step, workbench.runs);
  const document = workbench.document ?? initial.data ?? null;
  return (
    <div className="grid gap-4">
      <nav aria-label="Steps" className="flex flex-wrap gap-1">
        {STEPS.map((item, index) => (
          <Link
            key={item.id}
            to={{ name: "pipelines", step: item.id }}
            current={item.id === step}
            className={`rounded-md px-3 py-1 text-sm ${
              item.id === step
                ? "bg-accent text-accent-foreground"
                : "bg-default"
            }`}
          >
            {index + 1}. {item.title}
          </Link>
        ))}
      </nav>
      <Section
        title={definition.title}
        description={
          definition.until === null
            ? "Search the indexes of the latest run to Embed."
            : "Settings save as a new version of the pipeline spec when you run."
        }
        actions={
          <form
            className="flex flex-wrap items-end gap-2"
            onSubmit={(event) => {
              event.preventDefault();
              void load(nameDraft.trim());
            }}
          >
            <label className="grid gap-1 text-xs font-medium">
              Pipeline
              <input
                className={inputClass}
                value={nameDraft}
                onChange={(event) => setNameDraft(event.target.value)}
                pattern="[a-z0-9][a-z0-9._-]{0,63}"
              />
            </label>
            <Button type="submit" size="sm" variant="secondary">
              Load
            </Button>
            <Button
              size="sm"
              variant="tertiary"
              onPress={() => void load(workbench.specName, true)}
            >
              Reset to baseline
            </Button>
          </form>
        }
      >
        <ErrorNote error={error ?? initial.error} />
        {definition.until !== null ? (
          <div className="mb-4 flex flex-wrap items-end gap-2">
            <label className="grid gap-1 text-xs font-medium">
              Corpus version
              <select
                className={inputClass}
                value={corpusVersionId ?? ""}
                onChange={(event) =>
                  update({ corpusVersionId: event.target.value })
                }
              >
                {(versions.data ?? []).map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="grid gap-1 text-xs font-medium">
              Sample
              <select
                className={inputClass}
                value={
                  workbench.sampleSize === null
                    ? "all"
                    : String(workbench.sampleSize)
                }
                onChange={(event) =>
                  update({
                    sampleSize:
                      event.target.value === "all"
                        ? null
                        : Number(event.target.value),
                  })
                }
              >
                {SAMPLE_SIZES.map((size) => (
                  <option
                    key={String(size)}
                    value={size === null ? "all" : String(size)}
                  >
                    {size === null ? "All documents" : `${size} documents`}
                  </option>
                ))}
              </select>
            </label>
            <Button
              size="sm"
              isDisabled={busy || !document || !corpusVersionId}
              onPress={() => void runToStep()}
            >
              Run to this step
            </Button>
            {versions.data && versions.data.length === 0 ? (
              <span className="text-xs text-muted">
                Freeze a corpus version in{" "}
                <Link to={{ name: "documents" }} className="text-link">
                  Documents
                </Link>{" "}
                first.
              </span>
            ) : null}
          </div>
        ) : null}
        <div className="grid gap-4 lg:grid-cols-[22rem_1fr]">
          <div className="grid content-start gap-3" aria-label="Configure">
            {document === null || components.data === undefined ? (
              <Loading />
            ) : step === "search" ? (
              <Empty>Search uses the indexes of the latest run to Embed.</Empty>
            ) : (
              <>
                {step === "extract" ? <ExtractSources /> : null}
                {step === "embed" ? (
                  <ModelCatalog
                    runId={runFor("chunk", workbench.runs)?.runId ?? null}
                    document={document}
                    onChange={(next) => update({ document: next })}
                  />
                ) : null}
                {step === "parse" ? (
                  <ParseRoutes
                    document={document}
                    components={components.data ?? []}
                    onChange={(next) => update({ document: next })}
                  />
                ) : null}
                {slotsFor(step, document).map((slot) => (
                  <SlotEditor
                    key={slot.path}
                    slot={slot}
                    document={document}
                    components={components.data ?? []}
                    onChange={(next) => update({ document: next })}
                  />
                ))}
                {step === "scan" ? (
                  <>
                    <PolicyEditor
                      document={document}
                      onChange={(next) => update({ document: next })}
                    />
                    <RuleList
                      document={document}
                      components={components.data ?? []}
                      onChange={(next) => update({ document: next })}
                    />
                  </>
                ) : null}
              </>
            )}
          </div>
          <div className="grid content-start gap-3" aria-label="Results">
            {shown === null ? (
              <Empty>
                {step === "search"
                  ? "Run the pipeline to Embed first."
                  : "Run to this step to see its results."}
              </Empty>
            ) : step === "search" ? (
              <SearchPanel key={shown.runId} runId={shown.runId} />
            ) : (
              <Inspector
                key={shown.runId}
                step={step}
                runId={shown.runId}
                workbench={{
                  specName: workbench.specName,
                  document,
                  corpusVersionId,
                }}
              />
            )}
          </div>
        </div>
      </Section>
    </div>
  );
}
