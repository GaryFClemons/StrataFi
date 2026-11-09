import { Button } from "@heroui/react";
import { useState, type FormEvent } from "react";
import type { components } from "../api";
import { ApiError, unwrap, useLoad } from "../api/load";
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
import { formatBytes, formatTime, shortHash } from "../format";

type CorpusOut = components["schemas"]["CorpusOut"];
type EntryOut = components["schemas"]["EntryOut"];
type VersionOut = components["schemas"]["VersionOut"];

export function Documents() {
  const { api } = useLab();
  const navigate = useNavigate();
  const corpora = useLoad(
    async () =>
      unwrap(await api.GET("/corpora", { params: { query: { limit: 100 } } })),
    "corpora",
  );
  const [name, setName] = useState("");
  const [error, setError] = useState<unknown>(null);

  const create = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    try {
      const corpus = unwrap(
        await api.POST("/corpora", {
          body: { name: name.trim(), description: "" },
        }),
      );
      navigate({ name: "corpus", corpusId: corpus.id });
    } catch (failure) {
      setError(failure);
    }
  };

  return (
    <div className="grid gap-4">
      <Section
        title="Corpora"
        description="Each corpus is a working set of documents; freeze it to run on a version."
      >
        <form onSubmit={create} className="mb-3 flex flex-wrap items-end gap-2">
          <label className="grid gap-1 text-xs font-medium">
            New corpus
            <input
              className={inputClass}
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="lowercase-name"
              pattern="[a-z0-9][a-z0-9._\-]{0,63}"
              required
            />
          </label>
          <Button type="submit" size="sm">
            Create corpus
          </Button>
        </form>
        <ErrorNote error={error} />
        {corpora.data === undefined ? (
          corpora.error ? (
            <ErrorNote error={corpora.error} />
          ) : (
            <Loading />
          )
        ) : corpora.data.items.length === 0 ? (
          <Empty>No corpora yet.</Empty>
        ) : (
          <DataTable<CorpusOut>
            label="Corpora"
            rows={corpora.data.items}
            rowKey={(corpus) => corpus.id}
            columns={[
              {
                header: "Name",
                cell: (corpus) => (
                  <Link
                    to={{ name: "corpus", corpusId: corpus.id }}
                    className="text-link"
                  >
                    {corpus.name}
                  </Link>
                ),
              },
              {
                header: "Updated",
                cell: (corpus) => formatTime(corpus.updated_at),
              },
              { header: "By", cell: (corpus) => corpus.created_by },
            ]}
          />
        )}
      </Section>
    </div>
  );
}

async function upload(
  apiBase: string,
  headers: Record<string, string>,
  corpusId: string,
  files: File[],
): Promise<components["schemas"]["IngestOut"]> {
  const body = new FormData();
  for (const file of files) body.append("file", file, file.name);
  const response = await fetch(
    `${apiBase.replace(/\/+$/, "")}/documents?corpus_id=${encodeURIComponent(corpusId)}`,
    { method: "POST", headers, body },
  );
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(
      response.status,
      payload as components["schemas"]["ProblemDetails"],
    );
  }
  return payload as components["schemas"]["IngestOut"];
}

export function Corpus({ corpusId }: { corpusId: string }) {
  const { api, host } = useLab();
  const path = { params: { path: { corpus_id: corpusId } } };
  const detail = useLoad(
    async () => unwrap(await api.GET("/corpora/{corpus_id}", path)),
    corpusId,
  );
  const entries = useLoad(
    async () =>
      unwrap(
        await api.GET("/corpora/{corpus_id}/documents", {
          params: { path: { corpus_id: corpusId }, query: { limit: 200 } },
        }),
      ),
    corpusId,
  );
  const versions = useLoad(
    async () => unwrap(await api.GET("/corpora/{corpus_id}/versions", path)),
    corpusId,
  );
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const refresh = () => {
    detail.reload();
    entries.reload();
    versions.reload();
  };

  const send = async () => {
    setBusy(true);
    setError(null);
    try {
      const result = await upload(
        host.apiBase,
        await host.authHeaders(),
        corpusId,
        files,
      );
      host.notify(
        `${result.counts.files ?? files.length} file(s) uploaded, ${result.added_to_corpus} added.`,
        "success",
      );
      setFiles([]);
      refresh();
    } catch (failure) {
      setError(failure);
    } finally {
      setBusy(false);
    }
  };

  const freeze = async () => {
    setBusy(true);
    setError(null);
    try {
      const version = unwrap(
        await api.POST("/corpora/{corpus_id}/versions", path),
      );
      host.notify(`Version ${version.version} frozen.`, "success");
      refresh();
    } catch (failure) {
      setError(failure);
    } finally {
      setBusy(false);
    }
  };

  const corpus = detail.data;
  return (
    <div className="grid gap-4">
      <p className="text-sm">
        <Link to={{ name: "documents" }} className="text-link">
          Corpora
        </Link>{" "}
        / {corpus?.name ?? "…"}
      </p>
      <ErrorNote error={detail.error} />
      <Section
        title={corpus ? `Corpus ${corpus.name}` : "Corpus"}
        description={
          corpus
            ? `${corpus.entries} documents, ${formatBytes(corpus.total_bytes)}` +
              (corpus.duplicate_groups.length
                ? `; ${corpus.duplicate_groups.length} duplicate group(s)`
                : "")
            : undefined
        }
        actions={
          <Button
            size="sm"
            variant="secondary"
            isDisabled={busy}
            onPress={freeze}
          >
            Freeze version
          </Button>
        }
      >
        <div className="mb-3 flex flex-wrap items-end gap-2">
          <label className="grid gap-1 text-xs font-medium">
            Upload documents
            <input
              type="file"
              multiple
              className="text-sm"
              onChange={(event) =>
                setFiles(Array.from(event.target.files ?? []))
              }
            />
          </label>
          <Button
            size="sm"
            isDisabled={busy || files.length === 0}
            onPress={send}
          >
            Upload
          </Button>
        </div>
        <ErrorNote error={error} />
        {entries.data === undefined ? (
          <Loading />
        ) : entries.data.items.length === 0 ? (
          <Empty>No documents yet: upload some.</Empty>
        ) : (
          <DataTable<EntryOut>
            label="Documents"
            rows={entries.data.items}
            rowKey={(entry) => entry.name}
            columns={[
              { header: "Name", cell: (entry) => entry.name },
              { header: "Type", cell: (entry) => entry.media_type },
              {
                header: "Size",
                cell: (entry) => formatBytes(entry.size_bytes),
              },
              {
                header: "Traits",
                cell: (entry) =>
                  Object.entries(entry.traits)
                    .map(([key, value]) => `${key}: ${String(value)}`)
                    .join(", ") || "–",
              },
              {
                header: "SHA-256",
                cell: (entry) => <code>{shortHash(entry.sha256)}</code>,
              },
              {
                header: "Status",
                cell: (entry) => <StatusChip status={entry.status} />,
              },
            ]}
          />
        )}
      </Section>
      <Section
        title="Versions"
        description="Frozen versions never change; runs use them."
      >
        {versions.data === undefined ? (
          <Loading />
        ) : versions.data.items.length === 0 ? (
          <Empty>No versions yet: freeze the working set.</Empty>
        ) : (
          <DataTable<VersionOut>
            label="Versions"
            rows={versions.data.items}
            rowKey={(version) => version.id}
            columns={[
              { header: "Version", cell: (version) => `v${version.version}` },
              {
                header: "Documents",
                cell: (version) => version.document_count,
              },
              {
                header: "Size",
                cell: (version) => formatBytes(version.total_bytes),
              },
              {
                header: "Manifest",
                cell: (version) => (
                  <code>{shortHash(version.manifest_hash)}</code>
                ),
              },
              {
                header: "Frozen",
                cell: (version) => formatTime(version.frozen_at),
              },
            ]}
          />
        )}
      </Section>
    </div>
  );
}
