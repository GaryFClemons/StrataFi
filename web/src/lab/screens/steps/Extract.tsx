// The Extract step: where documents come
// from, and what each one is. Sources import into a corpus's working set:
// a URL through the SSRF guard (extract/url), or a configured bucket prefix
// (extract/s3); uploads stay on the Documents screen. The inspector shows
// each document's detected type against its extension, its family and
// traits, its hash, duplicate groups, and the tree of archive and email
// members. Mismatched and unsupported files are highlighted.

import { Button } from "@heroui/react";
import { useState } from "react";
import type { components } from "../../api";
import { unwrap, useLoad } from "../../api/load";
import type { RunOut } from "../../components/live";
import {
  DataTable,
  Empty,
  ErrorNote,
  StatusChip,
  inputClass,
} from "../../components/ui";
import { Link, useLab } from "../../context";
import { formatBytes, shortHash } from "../../format";
import { duplicateGroups, entryTree, type TreeNode } from "../../inspect";
import {
  Tasks,
  TaskStatus,
  summaryOf,
  useRunEntries,
  value,
  type Summary,
  type TaskOut,
} from "./shared";

type FileTypesOut = components["schemas"]["FileTypesOut"];
type IngestOut = components["schemas"]["IngestOut"];

export function useFileTypes() {
  const { api } = useLab();
  return useLoad(
    async () => unwrap(await api.GET("/file-types")),
    "file-types",
  );
}

function familyOf(types: FileTypesOut | undefined, mediaType: string): string {
  return (
    types?.types.find((item) => item.media_type === mediaType)?.family ??
    "other"
  );
}

function traitText(traits: Summary): string {
  return (
    Object.entries(traits)
      .filter(([, item]) => item !== false && item !== null)
      .map(([key, item]) => (item === true ? key : `${key} ${String(item)}`))
      .join(", ") || "–"
  );
}

/** Import into a corpus's working set from a URL or a configured bucket. */
export function ExtractSources() {
  const { api, host } = useLab();
  const sources = useLoad(
    async () => unwrap(await api.GET("/intake/sources")),
    "intake-sources",
  );
  const corpora = useLoad(
    async () =>
      unwrap(await api.GET("/corpora", { params: { query: { limit: 100 } } }))
        .items,
    "corpora",
  );
  const [corpusId, setCorpusId] = useState<string>("");
  const [url, setUrl] = useState("");
  const [bucket, setBucket] = useState("");
  const [prefix, setPrefix] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [result, setResult] = useState<IngestOut | null>(null);
  const corpus = corpusId || corpora.data?.[0]?.id || "";

  const run = async (call: () => Promise<IngestOut>) => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const report = await call();
      setResult(report);
      host.notify(
        `Imported ${report.counts.documents ?? 0} documents; freeze a new version to run on them.`,
        "success",
      );
    } catch (failure) {
      setError(failure);
    } finally {
      setBusy(false);
    }
  };

  return (
    <fieldset
      className="grid gap-2 rounded-md border border-separator p-3"
      aria-label="Sources"
    >
      <legend className="px-1 text-xs font-semibold">Sources</legend>
      <label className="grid gap-1 text-xs font-medium">
        Into corpus
        <select
          className={inputClass}
          value={corpus}
          onChange={(event) => setCorpusId(event.target.value)}
        >
          {(corpora.data ?? []).map((item) => (
            <option key={item.id} value={item.id}>
              {item.name}
            </option>
          ))}
        </select>
      </label>
      {sources.data?.url_import ? (
        <form
          className="grid gap-1"
          onSubmit={(event) => {
            event.preventDefault();
            void run(async () =>
              unwrap(
                await api.POST("/documents/import-url", {
                  body: { url: url.trim(), corpus_id: corpus || null },
                }),
              ),
            );
          }}
        >
          <label className="grid gap-1 text-xs font-medium">
            Fetch a URL
            <input
              className={inputClass}
              type="url"
              value={url}
              onChange={(event) => setUrl(event.target.value)}
              placeholder="https://"
              required
            />
          </label>
          <span className="text-xs text-muted">
            Public addresses only, checked again on every redirect.
          </span>
          <Button type="submit" size="sm" isDisabled={busy || !corpus}>
            Fetch
          </Button>
        </form>
      ) : null}
      {sources.data && sources.data.s3_sources.length > 0 ? (
        <form
          className="grid gap-1"
          onSubmit={(event) => {
            event.preventDefault();
            void run(async () =>
              unwrap(
                await api.POST("/documents/import-s3", {
                  body: {
                    source: bucket || sources.data!.s3_sources[0]!,
                    prefix,
                    corpus_id: corpus || null,
                  },
                }),
              ),
            );
          }}
        >
          <label className="grid gap-1 text-xs font-medium">
            Bucket
            <select
              className={inputClass}
              value={bucket || sources.data.s3_sources[0]}
              onChange={(event) => setBucket(event.target.value)}
            >
              {sources.data.s3_sources.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          <label className="grid gap-1 text-xs font-medium">
            Prefix
            <input
              className={inputClass}
              value={prefix}
              onChange={(event) => setPrefix(event.target.value)}
            />
          </label>
          <Button type="submit" size="sm" isDisabled={busy || !corpus}>
            Import bucket
          </Button>
        </form>
      ) : null}
      <span className="text-xs text-muted">
        Upload files on the{" "}
        <Link to={{ name: "documents" }} className="text-link">
          Documents
        </Link>{" "}
        screen. Archives and email unpack into their members.
      </span>
      <ErrorNote error={error ?? sources.error ?? corpora.error} />
      {result ? (
        <p className="text-xs" role="status">
          {Object.entries(result.counts)
            .map(([key, count]) => `${key} ${count}`)
            .join(" · ")}
        </p>
      ) : null}
    </fieldset>
  );
}

function Tree({
  nodes,
  tasks,
}: {
  nodes: TreeNode[];
  tasks: Map<string, TaskOut>;
}) {
  return (
    <ul className="grid gap-0.5 pl-4 text-sm">
      {nodes.map((node) => {
        const task = node.entry ? tasks.get(node.entry) : undefined;
        const summary = task ? summaryOf(task) : {};
        return (
          <li key={`${node.kind}:${node.entry ?? node.label}`}>
            <span className="flex flex-wrap items-center gap-2">
              <span aria-hidden="true">
                {node.kind === "container"
                  ? "▣"
                  : node.kind === "folder"
                    ? "▸"
                    : "·"}
              </span>
              <span className={node.kind === "file" ? "" : "font-medium"}>
                {node.label}
              </span>
              {node.kind === "container" ? (
                <span className="text-xs text-muted">
                  {node.children.length} members
                </span>
              ) : null}
              {summary.mismatch === true ? (
                <StatusChip status="flag" label="mismatch" />
              ) : null}
            </span>
            {node.children.length ? (
              <Tree nodes={node.children} tasks={tasks} />
            ) : null}
          </li>
        );
      })}
    </ul>
  );
}

export function ExtractInspector({ run }: { run: RunOut }) {
  const types = useFileTypes();
  const entries = useRunEntries(run);
  return (
    <Tasks run={run} stage="extract" empty="No documents extracted yet.">
      {(tasks) => {
        const byEntry = new Map(
          tasks.filter((task) => task.entry).map((task) => [task.entry!, task]),
        );
        const hashes = new Map(
          (entries.data ?? []).map((entry) => [entry.name, entry.sha256]),
        );
        const groups = duplicateGroups(entries.data ?? []);
        const flagged = tasks.filter(
          (task) =>
            summaryOf(task).mismatch === true ||
            summaryOf(task).known === false,
        ).length;
        return (
          <div className="grid gap-3">
            <p className="text-xs text-muted" aria-label="Extract summary">
              {tasks.length} documents · {flagged} mismatched or unlisted ·{" "}
              {groups.length} duplicate groups
            </p>
            <DataTable<TaskOut>
              label="Extracted documents"
              rows={tasks}
              rowKey={(task) => task.id}
              columns={[
                { header: "Document", cell: (task) => task.entry ?? "–" },
                {
                  header: "Family",
                  cell: (task) =>
                    familyOf(
                      types.data,
                      String(summaryOf(task).media_type ?? ""),
                    ),
                },
                {
                  header: "Detected type",
                  cell: (task) => value(summaryOf(task), "media_type"),
                },
                {
                  header: "Extension",
                  cell: (task) => value(summaryOf(task), "extension"),
                },
                {
                  header: "Matches",
                  cell: (task) =>
                    summaryOf(task).mismatch === true ? (
                      <StatusChip status="flag" label="mismatch" />
                    ) : summaryOf(task).known === false ? (
                      <StatusChip status="flag" label="unlisted type" />
                    ) : (
                      "yes"
                    ),
                },
                {
                  header: "Traits",
                  cell: (task) =>
                    traitText((summaryOf(task).traits ?? {}) as Summary),
                },
                {
                  header: "Hash",
                  cell: (task) => (
                    <code>{shortHash(hashes.get(task.entry ?? ""), 10)}</code>
                  ),
                },
                {
                  header: "Size",
                  cell: (task) =>
                    formatBytes(
                      summaryOf(task).size_bytes as number | undefined,
                    ),
                },
                {
                  header: "Status",
                  cell: (task) => <TaskStatus task={task} />,
                },
              ]}
            />
            <section aria-label="Duplicate groups" className="grid gap-1">
              <h3 className="text-sm font-semibold">Duplicate groups</h3>
              {groups.length === 0 ? (
                <Empty>No two documents share bytes.</Empty>
              ) : (
                <ul className="grid gap-1 text-sm">
                  {groups.map((group) => (
                    <li key={group[0]}>
                      <span className="font-medium">{group[0]}</span>
                      <span className="text-muted">
                        {" "}
                        (kept) · {group.slice(1).join(", ")}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </section>
            <section
              aria-label="Archive and attachment tree"
              className="grid gap-1"
            >
              <h3 className="text-sm font-semibold">
                Archives and attachments
              </h3>
              <Tree
                nodes={entryTree(
                  (entries.data ?? []).map((entry) => entry.name),
                )}
                tasks={byEntry}
              />
            </section>
          </div>
        );
      }}
    </Tasks>
  );
}
