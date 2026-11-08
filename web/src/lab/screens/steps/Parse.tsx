// The Parse step. Configure: the routing table
// (file type, with traits, → parser and settings). Results: each document's
// parse with its quality metrics, and a viewer: the original page rendered by
// pdf.js in a worker beside the parsed elements, coloured by type, with
// element boxes drawn over the page when the parser gives them, both kept on
// the same page; a table viewer; 2–3 parsers compared on one page; and "ask a
// question about this passage", which adds an eval item.

import { Button } from "@heroui/react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { components } from "../../api";
import { unwrap, useLoad } from "../../api/load";
import { useLiveRun, type RunOut } from "../../components/live";
import { SlotEditor } from "../../components/SlotEditor";
import {
  DataTable,
  Empty,
  ErrorNote,
  Loading,
  StatusChip,
  inputClass,
} from "../../components/ui";
import { useLab } from "../../context";
import {
  colourOf,
  elementsOnPage,
  evidenceOf,
  overlayRect,
  pageNumbers,
  tableRows,
  type ParsedDocument,
  type ParsedElement,
} from "../../parse";
import type { PdfDocument } from "../../pdf";
import { getAt, setAt, type SpecDocument } from "../../spec/editing";
import { useFileTypes } from "./Extract";
import {
  Tasks,
  TaskStatus,
  summaryOf,
  useArtifactJson,
  value,
  type TaskOut,
} from "./shared";

type ComponentOut = components["schemas"]["ComponentOut"];
type PipelineSpec = components["schemas"]["PipelineSpec"];

export interface Workbench {
  specName: string;
  document: SpecDocument | null;
  corpusVersionId: string | null;
}

// --- the routing table ----------------------------------------------------------------------

export function ParseRoutes({
  document,
  components,
  onChange,
}: {
  document: SpecDocument;
  components: ComponentOut[];
  onChange: (document: SpecDocument) => void;
}) {
  const types = useFileTypes();
  const route = (getAt(document, "stages.parse.route") ?? {}) as Record<
    string,
    unknown
  >;
  const [key, setKey] = useState("");
  const parsers = components.filter((item) => item.stage === "parse");
  const suggestions = useMemo(() => {
    const shorts = (types.data?.types ?? [])
      .filter((item) => item.family !== "other" && item.family !== "archive")
      .map((item) => item.short);
    const flags = (types.data?.traits ?? []).filter(
      (trait) => trait.kind === "flag",
    );
    const traited = flags.flatMap((trait) =>
      trait.applies_to.map((short) => `${short}[${trait.name}]`),
    );
    return [...shorts, ...traited];
  }, [types.data]);
  return (
    <div className="grid gap-2" aria-label="Routing table">
      <p className="text-xs font-semibold">Routing table</p>
      <p className="text-xs text-muted">
        Each file type, or type with traits, goes to a parser. The most specific
        matching route wins; the rest go to the default.
      </p>
      {Object.keys(route)
        .sort((a, b) =>
          a === "default" ? -1 : b === "default" ? 1 : a.localeCompare(b),
        )
        .map((routeKey) => (
          <SlotEditor
            key={routeKey}
            slot={{
              path: `stages.parse.route.${routeKey}`,
              stage: "parse",
              label:
                routeKey === "default" ? "Default parser" : `Route ${routeKey}`,
            }}
            document={document}
            components={components}
            onChange={onChange}
            actions={
              routeKey === "default" ? undefined : (
                <Button
                  size="sm"
                  variant="tertiary"
                  aria-label={`Remove route ${routeKey}`}
                  onPress={() =>
                    onChange(
                      setAt(
                        document,
                        `stages.parse.route.${routeKey}`,
                        undefined,
                      ),
                    )
                  }
                >
                  Remove
                </Button>
              )
            }
          />
        ))}
      <form
        className="flex flex-wrap items-end gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          const routeKey = key.trim();
          if (!routeKey || routeKey in route) return;
          const parser =
            parsers.find((item) => item.id === "parse/pdfplumber") ??
            parsers[0];
          if (!parser) return;
          onChange(
            setAt(document, `stages.parse.route.${routeKey}`, {
              component: parser.id,
              params: {},
            }),
          );
          setKey("");
        }}
      >
        <label className="grid gap-1 text-xs font-medium">
          Add a route
          <input
            className={inputClass}
            list="route-keys"
            value={key}
            onChange={(event) => setKey(event.target.value)}
            placeholder="pdf[scanned]"
            pattern="[a-z0-9][a-z0-9.+/-]*(\[[a-z_,]+\])?"
          />
        </label>
        <datalist id="route-keys">
          {suggestions.map((item) => (
            <option key={item} value={item} />
          ))}
        </datalist>
        <Button type="submit" size="sm" variant="secondary">
          Add route
        </Button>
      </form>
    </div>
  );
}

// --- the viewer -------------------------------------------------------------------------------

function useOriginal(runId: string, ordinal: number, mediaType: string) {
  const { api } = useLab();
  return useLoad<ArrayBuffer | null>(async () => {
    if (!["application/pdf", "image/png", "image/jpeg"].includes(mediaType))
      return null;
    const result = await api.GET("/runs/{run_id}/entries/{ordinal}/original", {
      params: { path: { run_id: runId, ordinal } },
      parseAs: "arrayBuffer",
    });
    return unwrap(result) as unknown as ArrayBuffer;
  }, `${runId}:${ordinal}:original:${mediaType}`);
}

function PageView({
  original,
  mediaType,
  page,
  elements,
  pageHeight,
  selected,
  onSelect,
}: {
  original: ArrayBuffer;
  mediaType: string;
  page: number;
  elements: ParsedElement[];
  pageHeight: number | null;
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [pdf, setPdf] = useState<PdfDocument | null>(null);
  const [size, setSize] = useState<{ width: number; height: number } | null>(
    null,
  );
  const [error, setError] = useState<unknown>(null);
  const scale = 1.2;
  const image = useMemo(
    () =>
      mediaType === "application/pdf"
        ? null
        : URL.createObjectURL(new Blob([original], { type: mediaType })),
    [original, mediaType],
  );
  useEffect(
    () => () => {
      if (image) URL.revokeObjectURL(image);
    },
    [image],
  );

  useEffect(() => {
    if (mediaType !== "application/pdf") return;
    let alive = true;
    // pdf.js loads with the first PDF, so the rest of the lab doesn't carry it.
    import("../../pdf")
      .then(({ openPdf }) => openPdf(original.slice(0)))
      .then((opened) => alive && setPdf(opened))
      .catch((failure: unknown) => alive && setError(failure));
    return () => {
      alive = false;
    };
  }, [original, mediaType]);

  useEffect(() => {
    if (!pdf || !canvas.current) return;
    let alive = true;
    const target = canvas.current;
    import("../../pdf")
      .then(({ renderPage }) => renderPage(pdf, page, target, scale))
      .then((found) => alive && setSize(found))
      .catch((failure: unknown) => alive && setError(failure));
    return () => {
      alive = false;
    };
  }, [pdf, page]);

  const height = size?.height ?? pageHeight ?? 0;
  return (
    <div className="grid gap-1">
      {error ? <ErrorNote error={error} /> : null}
      <div
        className="relative inline-block max-w-full overflow-auto rounded-md border border-separator"
        data-testid="page-view"
      >
        {image ? (
          <img
            src={image}
            alt={`Page ${page} of the original`}
            className="block max-w-none"
            style={{
              transform: `scale(${scale})`,
              transformOrigin: "top left",
            }}
          />
        ) : (
          <canvas
            ref={canvas}
            aria-label={`Page ${page} of the original`}
            data-testid="pdf-page"
            data-rendered={size ? "true" : "false"}
          />
        )}
        {height
          ? elements
              .filter((element) => element.bbox)
              .map((element) => {
                const rect = overlayRect(element.bbox!, height, scale);
                return (
                  <button
                    type="button"
                    key={element.id}
                    data-testid="element-box"
                    aria-label={`${element.type} ${element.id}`}
                    onClick={() => onSelect(element.id)}
                    className="absolute rounded-sm"
                    style={{
                      ...rect,
                      border: `2px solid ${colourOf(element.type)}`,
                      background:
                        selected === element.id
                          ? `${colourOf(element.type)}33`
                          : "transparent",
                    }}
                  />
                );
              })
          : null}
      </div>
    </div>
  );
}

function TableView({ element }: { element: ParsedElement }) {
  if (!element.table) return null;
  const rows = tableRows(element.table);
  return (
    <div className="overflow-x-auto" aria-label="Table viewer">
      <table className="border-collapse text-xs">
        <tbody>
          {rows.map((row, r) => (
            <tr key={r}>
              {row.map((cell, c) =>
                cell.header ? (
                  <th
                    key={c}
                    className="border border-separator px-2 py-1 text-left"
                  >
                    {cell.text}
                  </th>
                ) : (
                  <td key={c} className="border border-separator px-2 py-1">
                    {cell.text}
                  </td>
                ),
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AuthorForm({
  entry,
  elements,
  index,
  onDone,
}: {
  entry: string;
  elements: ParsedElement[];
  index: number;
  onDone: () => void;
}) {
  const { api, host } = useLab();
  const [query, setQuery] = useState("");
  const [evalSet, setEvalSet] = useState("authored");
  const [error, setError] = useState<unknown>(null);
  const evidence = evidenceOf(entry, elements, index);
  return (
    <form
      className="grid gap-1 rounded-md border border-separator p-2"
      aria-label="Ask a question about this passage"
      onSubmit={(event) => {
        event.preventDefault();
        setError(null);
        void (async () => {
          try {
            const saved = unwrap(
              await api.POST("/eval-sets/author", {
                body: {
                  eval_set: evalSet,
                  query: query.trim(),
                  evidence: {
                    doc_key: evidence.doc_key,
                    section: evidence.section,
                    evidence: evidence.evidence,
                  },
                  tags: [],
                },
              }),
            );
            host.notify(
              `Added ${saved.item_id} to ${saved.eval_set.name} v${saved.version.version}.`,
              "success",
            );
            onDone();
          } catch (failure) {
            setError(failure);
          }
        })();
      }}
    >
      <label className="grid gap-1 text-xs font-medium">
        Question
        <input
          className={inputClass}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          required
          minLength={3}
        />
      </label>
      <label className="grid gap-1 text-xs font-medium">
        Eval set
        <input
          className={inputClass}
          value={evalSet}
          onChange={(event) => setEvalSet(event.target.value)}
          pattern="[a-z0-9][a-z0-9._-]{0,63}"
        />
      </label>
      <div className="flex gap-2">
        <Button type="submit" size="sm">
          Add to eval set
        </Button>
        <Button size="sm" variant="tertiary" onPress={onDone}>
          Cancel
        </Button>
      </div>
      <ErrorNote error={error} />
    </form>
  );
}

function ElementList({
  entry,
  document,
  page,
  selected,
  onSelect,
  label = "Parsed elements",
  authoring = true,
}: {
  entry: string;
  document: ParsedDocument;
  page: number;
  selected: string | null;
  onSelect: (id: string) => void;
  label?: string;
  authoring?: boolean;
}) {
  const [asking, setAsking] = useState<string | null>(null);
  const shown = elementsOnPage(document, page);
  const refs = useRef(new Map<string, HTMLLIElement>());
  useEffect(() => {
    if (selected)
      refs.current.get(selected)?.scrollIntoView({ block: "nearest" });
  }, [selected]);
  return (
    <ol
      aria-label={label}
      className="grid max-h-[40rem] content-start gap-1 overflow-auto"
    >
      {shown.length === 0 ? <Empty>No elements on this page.</Empty> : null}
      {shown.map((element) => {
        const index = document.elements.indexOf(element);
        return (
          <li
            key={element.id}
            ref={(node) => {
              if (node) refs.current.set(element.id, node);
            }}
            data-testid="parsed-element"
            className="grid gap-1 rounded-md border-l-4 bg-default p-2 text-xs"
            style={{
              borderLeftColor: colourOf(element.type),
              outline:
                selected === element.id ? "2px solid currentColor" : undefined,
            }}
            onClick={() => onSelect(element.id)}
          >
            <span className="flex flex-wrap items-center gap-2 text-muted">
              <code>{element.type}</code>
              {element.level ? `level ${element.level}` : null}
              {element.page ? `page ${element.page}` : null}
            </span>
            {element.table ? (
              <TableView element={element} />
            ) : (
              <span className="whitespace-pre-wrap break-words">
                {element.text}
              </span>
            )}
            {authoring ? (
              asking === element.id ? (
                <AuthorForm
                  entry={entry}
                  elements={document.elements}
                  index={index}
                  onDone={() => setAsking(null)}
                />
              ) : (
                <button
                  type="button"
                  className="justify-self-start text-link"
                  onClick={(event) => {
                    event.stopPropagation();
                    setAsking(element.id);
                  }}
                >
                  Ask a question about this passage
                </button>
              )
            ) : null}
          </li>
        );
      })}
    </ol>
  );
}

function DocumentView({
  run,
  task,
  workbench,
}: {
  run: RunOut;
  task: TaskOut;
  workbench: Workbench;
}) {
  const parsed = useArtifactJson<ParsedDocument>(task.cache_key);
  const mediaType = parsed.data?.media_type ?? "";
  const original = useOriginal(run.id, task.ordinal, mediaType);
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState<string | null>(null);
  if (parsed.error) return <ErrorNote error={parsed.error} />;
  if (!parsed.data) return <Loading />;
  const document = parsed.data;
  const pages = pageNumbers(document);
  const current = pages.length ? (pages.includes(page) ? page : pages[0]!) : 0;
  const pageInfo = document.pages.find((item) => item.number === current);
  const types = Object.entries(
    document.elements.reduce<Record<string, number>>((counts, element) => {
      counts[element.type] = (counts[element.type] ?? 0) + 1;
      return counts;
    }, {}),
  );
  const move = (step: number) => {
    const index = pages.indexOf(current) + step;
    if (index >= 0 && index < pages.length) setPage(pages[index]!);
  };
  return (
    <section aria-label="Document viewer" className="grid gap-3">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        {pages.length ? (
          <>
            <Button
              size="sm"
              variant="tertiary"
              onPress={() => move(-1)}
              aria-label="Previous page"
            >
              ←
            </Button>
            <span aria-label="Page">
              Page {current} of {pages.length}
            </span>
            <Button
              size="sm"
              variant="tertiary"
              onPress={() => move(1)}
              aria-label="Next page"
            >
              →
            </Button>
          </>
        ) : null}
        {types.map(([type, count]) => (
          <span key={type} className="flex items-center gap-1">
            <span
              aria-hidden="true"
              className="inline-block h-3 w-3 rounded-sm"
              style={{ background: colourOf(type) }}
            />
            {type} {count}
          </span>
        ))}
        {document.warnings.map((warning) => (
          <StatusChip key={warning} status="flag" label={warning} />
        ))}
      </div>
      <div className="grid gap-3 xl:grid-cols-2">
        {original.data ? (
          <PageView
            original={original.data}
            mediaType={mediaType}
            page={current}
            elements={elementsOnPage(document, current)}
            pageHeight={pageInfo?.height ?? null}
            selected={selected}
            onSelect={setSelected}
          />
        ) : original.error ? (
          <ErrorNote error={original.error} />
        ) : null}
        <ElementList
          entry={task.entry ?? "document"}
          document={document}
          page={current}
          selected={selected}
          onSelect={setSelected}
        />
      </div>
      <ComparePanel
        run={run}
        task={task}
        workbench={workbench}
        page={current}
        mediaType={mediaType}
      />
    </section>
  );
}

// --- comparing parsers ----------------------------------------------------------------------

function CompareColumn({ runId, page }: { runId: string; page: number }) {
  const live = useLiveRun(runId);
  const { api } = useLab();
  const done = live.data?.status === "succeeded";
  const tasks = useLoad(
    async () =>
      done
        ? unwrap(
            await api.GET("/runs/{run_id}/tasks", {
              params: { path: { run_id: runId }, query: { stage: "parse" } },
            }),
          ).items
        : null,
    `${runId}:compare:${done}`,
  );
  const task = tasks.data?.[0] ?? null;
  const parsed = useArtifactJson<ParsedDocument>(task?.cache_key ?? null);
  if (live.error) return <ErrorNote error={live.error} />;
  if (
    live.data &&
    live.data.status !== "succeeded" &&
    live.data.status !== "running" &&
    live.data.status !== "queued"
  )
    return <StatusChip status={live.data.status} />;
  if (!parsed.data || !task)
    return <Loading label={`Parsing with ${live.data?.status ?? "…"}`} />;
  const summary = summaryOf(task);
  return (
    <div className="grid content-start gap-1" data-testid="compare-column">
      <p className="text-xs font-semibold">{task.component_id}</p>
      <p className="text-xs text-muted">
        {value(summary, "elements")} elements · {value(summary, "chars")}{" "}
        characters · {value(summary, "headings")} headings ·{" "}
        {value(summary, "tables")} tables
      </p>
      <ElementList
        entry={task.entry ?? "document"}
        document={parsed.data}
        page={page}
        selected={null}
        onSelect={() => undefined}
        label={`Elements from ${task.component_id}`}
        authoring={false}
      />
    </div>
  );
}

function ComparePanel({
  run,
  task,
  workbench,
  page,
  mediaType,
}: {
  run: RunOut;
  task: TaskOut;
  workbench: Workbench;
  page: number;
  mediaType: string;
}) {
  const { api } = useLab();
  const components = useLoad(
    async () =>
      unwrap(
        await api.GET("/components", { params: { query: { stage: "parse" } } }),
      ).items,
    "parse-components",
  );
  const [chosen, setChosen] = useState<string[]>([]);
  const [runs, setRuns] = useState<string[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const candidates = (components.data ?? []).filter(
    (item) =>
      item.available &&
      item.id !== task.component_id &&
      item.egress === "none" &&
      item.accepts.some(
        (pattern) =>
          pattern === mediaType || pattern.split("[")[0] === mediaType,
      ),
  );
  const compare = async () => {
    if (!workbench.document || !task.entry) return;
    setBusy(true);
    setError(null);
    try {
      const started: string[] = [];
      for (const parser of chosen) {
        const variant = setAt(
          workbench.document,
          `stages.parse.route.${task.route ?? "default"}`,
          {
            component: parser,
            params: {},
          },
        );
        const saved = unwrap(
          await api.POST("/pipeline-specs", {
            body: {
              ...variant,
              name: `${workbench.specName}-compare-${parser.split("/")[1]}`.slice(
                0,
                64,
              ),
            } as PipelineSpec,
          }),
        );
        const started_run = unwrap(
          await api.POST("/runs", {
            body: {
              pipeline_spec_id: saved.id,
              corpus_version_id: run.corpus_version_id,
              until: "parse",
              sample: { entries: [task.entry] },
              mode: "interactive",
            },
          }),
        );
        started.push(started_run.id);
      }
      setRuns(started);
    } catch (failure) {
      setError(failure);
    } finally {
      setBusy(false);
    }
  };
  return (
    <section
      aria-label="Compare parsers"
      className="grid gap-2 rounded-md border border-separator p-3"
    >
      <p className="text-xs font-semibold">Compare parsers on this page</p>
      <div className="flex flex-wrap items-center gap-3 text-xs">
        {candidates.length === 0 ? (
          <span className="text-muted">
            No other installed parser accepts this type.
          </span>
        ) : (
          candidates.map((item) => (
            <label key={item.id} className="flex items-center gap-1">
              <input
                type="checkbox"
                checked={chosen.includes(item.id)}
                disabled={!chosen.includes(item.id) && chosen.length >= 2}
                onChange={(event) =>
                  setChosen((current) =>
                    event.target.checked
                      ? [...current, item.id]
                      : current.filter((id) => id !== item.id),
                  )
                }
              />
              {item.title}
            </label>
          ))
        )}
        <Button
          size="sm"
          variant="secondary"
          isDisabled={busy || chosen.length === 0}
          onPress={() => void compare()}
        >
          Compare
        </Button>
      </div>
      <ErrorNote error={error ?? components.error} />
      {runs.length ? (
        <div
          className={`grid gap-3 ${runs.length > 1 ? "lg:grid-cols-2" : ""}`}
        >
          {runs.map((runId) => (
            <CompareColumn key={runId} runId={runId} page={page} />
          ))}
        </div>
      ) : null}
    </section>
  );
}

// --- the inspector --------------------------------------------------------------------------

function percent(value: unknown): string {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "–";
}

export function ParseInspector({
  run,
  workbench,
}: {
  run: RunOut;
  workbench: Workbench;
}) {
  const [chosen, setChosen] = useState<string | null>(null);
  return (
    <Tasks run={run} stage="parse" empty="No documents parsed yet.">
      {(tasks) => {
        const done = tasks.filter(
          (task) => summaryOf(task).chars !== undefined,
        );
        const shown =
          tasks.find((task) => task.cache_key === chosen) ?? done[0] ?? null;
        return (
          <div className="grid gap-3">
            <DataTable<TaskOut>
              label="Parsed documents"
              rows={tasks}
              rowKey={(task) => task.cache_key}
              selected={shown?.cache_key ?? null}
              onSelect={(task) => setChosen(task.cache_key)}
              columns={[
                { header: "Document", cell: (task) => task.entry ?? "–" },
                { header: "Route", cell: (task) => task.route ?? "–" },
                { header: "Parser", cell: (task) => task.component_id },
                {
                  header: "Pages",
                  cell: (task) => value(summaryOf(task), "pages"),
                },
                {
                  header: "Chars/page",
                  cell: (task) => value(summaryOf(task), "chars_per_page_p50"),
                },
                {
                  header: "Empty pages",
                  cell: (task) => percent(summaryOf(task).empty_page_rate),
                },
                {
                  header: "Headings",
                  cell: (task) => value(summaryOf(task), "headings"),
                },
                {
                  header: "Tables",
                  cell: (task) => value(summaryOf(task), "tables"),
                },
                {
                  header: "OCR pages",
                  cell: (task) => value(summaryOf(task), "ocr_pages"),
                },
                {
                  header: "Garbage",
                  cell: (task) => percent(summaryOf(task).garbage_char_ratio),
                },
                {
                  header: "Status",
                  cell: (task) => <TaskStatus task={task} />,
                },
              ]}
            />
            {shown && summaryOf(shown).chars !== undefined ? (
              <DocumentView
                key={shown.cache_key}
                run={run}
                task={shown}
                workbench={workbench}
              />
            ) : null}
          </div>
        );
      }}
    </Tasks>
  );
}
