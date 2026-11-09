import { Button } from "@heroui/react";
import { useState, type FormEvent } from "react";
import type { components } from "../api";
import { unwrap, useLoad } from "../api/load";
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
import { formatTime, shortHash } from "../format";
import { ReviewQueue } from "./Answers";

type EvalSetOut = components["schemas"]["EvalSetOut"];
type VersionOut = components["schemas"]["EvalSetVersionOut"];
type ItemOut = components["schemas"]["EvalItemOut"];
type Format = components["schemas"]["EvalSetImport"]["format"];

const FORMATS: { value: Format; label: string }[] = [
  {
    value: "yaml",
    label: "YAML (stratafi.eval-set.v1 or its retrieval format)",
  },
  { value: "jsonl", label: "JSONL (one item per line)" },
  { value: "csv", label: "CSV (one row per evidence item)" },
];

function formatOf(fileName: string): Format {
  if (fileName.endsWith(".jsonl")) return "jsonl";
  if (fileName.endsWith(".csv")) return "csv";
  return "yaml";
}

function splitSummary(version: VersionOut | null | undefined): string {
  if (!version) return "–";
  const counts = version.split_counts;
  return `${counts.dev ?? 0} dev / ${counts.test ?? 0} test`;
}

/** Eval sets and the import wizard. */
export function EvalSets() {
  const { api, host } = useLab();
  const navigate = useNavigate();
  const sets = useLoad(
    async () =>
      unwrap(
        await api.GET("/eval-sets", { params: { query: { limit: 100 } } }),
      ),
    "eval-sets",
  );
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [format, setFormat] = useState<Format>("yaml");
  const [testFraction, setTestFraction] = useState(0.3);
  const [seed, setSeed] = useState(0);
  const [freeze, setFreeze] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const send = async (event: FormEvent) => {
    event.preventDefault();
    if (file === null) return;
    setBusy(true);
    setError(null);
    try {
      const imported = unwrap(
        await api.POST("/eval-sets/import", {
          body: {
            name: name.trim(),
            format,
            content: await file.text(),
            test_fraction: testFraction,
            seed,
            freeze,
            resplit: false,
          },
        }),
      );
      host.notify(
        imported.created
          ? `Version ${imported.version.version} imported: ${imported.report.items} items.`
          : `Unchanged: version ${imported.version.version} holds this set.`,
        "success",
      );
      navigate({ name: "evalset", evalSetId: imported.eval_set.id });
    } catch (failure) {
      setError(failure);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid gap-4">
      <Section
        title="Eval sets"
        description="Questions with known evidence. Tune on the dev split; the test split is locked, and each read of it is counted."
      >
        <form
          onSubmit={send}
          className="mb-3 flex flex-wrap items-end gap-2"
          aria-label="Import an eval set"
        >
          <label className="grid gap-1 text-xs font-medium">
            Eval set name
            <input
              className={inputClass}
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="halcyon-retrieval"
              pattern="[a-z0-9][a-z0-9._\-]{0,63}"
              required
            />
          </label>
          <label className="grid gap-1 text-xs font-medium">
            Eval set file
            <input
              type="file"
              accept=".yaml,.yml,.jsonl,.csv"
              className="text-sm"
              onChange={(event) => {
                const chosen = event.target.files?.[0] ?? null;
                setFile(chosen);
                if (chosen) setFormat(formatOf(chosen.name));
              }}
              required
            />
          </label>
          <label className="grid gap-1 text-xs font-medium">
            Format
            <select
              className={inputClass}
              value={format}
              onChange={(event) => setFormat(event.target.value as Format)}
            >
              {FORMATS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label className="grid gap-1 text-xs font-medium">
            Test share
            <input
              type="number"
              className={`${inputClass} w-20`}
              min={0}
              max={0.9}
              step={0.05}
              value={testFraction}
              onChange={(event) => setTestFraction(Number(event.target.value))}
            />
          </label>
          <label className="grid gap-1 text-xs font-medium">
            Split seed
            <input
              type="number"
              className={`${inputClass} w-24`}
              min={0}
              value={seed}
              onChange={(event) => setSeed(Number(event.target.value))}
            />
          </label>
          <label className="flex items-center gap-1 text-xs font-medium">
            <input
              type="checkbox"
              checked={freeze}
              onChange={(event) => setFreeze(event.target.checked)}
            />
            Freeze at once
          </label>
          <Button type="submit" size="sm" isDisabled={busy || file === null}>
            Import
          </Button>
        </form>
        <ErrorNote error={error} />
        {sets.data === undefined ? (
          sets.error ? (
            <ErrorNote error={sets.error} />
          ) : (
            <Loading />
          )
        ) : sets.data.items.length === 0 ? (
          <Empty>No eval sets yet: import one.</Empty>
        ) : (
          <DataTable<EvalSetOut>
            label="Eval sets"
            rows={sets.data.items}
            rowKey={(set) => set.id}
            columns={[
              {
                header: "Name",
                cell: (set) => (
                  <Link
                    to={{ name: "evalset", evalSetId: set.id }}
                    className="text-link"
                  >
                    {set.name}
                  </Link>
                ),
              },
              {
                header: "Latest",
                cell: (set) =>
                  set.latest_version ? `v${set.latest_version.version}` : "–",
              },
              {
                header: "Items",
                cell: (set) => set.latest_version?.item_count ?? "–",
              },
              {
                header: "Splits",
                cell: (set) => splitSummary(set.latest_version),
              },
              {
                header: "Test reads",
                cell: (set) => set.latest_version?.test_reads ?? "–",
              },
              { header: "Updated", cell: (set) => formatTime(set.updated_at) },
            ]}
          />
        )}
      </Section>
    </div>
  );
}

export function EvalSetDetail({ evalSetId }: { evalSetId: string }) {
  const { api, host } = useLab();
  const path = { params: { path: { eval_set_id: evalSetId } } };
  const detail = useLoad(
    async () => unwrap(await api.GET("/eval-sets/{eval_set_id}", path)),
    evalSetId,
  );
  const versions = useLoad(
    async () =>
      unwrap(await api.GET("/eval-sets/{eval_set_id}/versions", path)),
    evalSetId,
  );
  const [chosen, setChosen] = useState<number | null>(null);
  const [split, setSplit] = useState<"" | "dev" | "test">("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const version =
    versions.data?.items.find((item) => item.version === chosen) ??
    versions.data?.items[0] ??
    null;
  const items = useLoad(
    async () =>
      version === null
        ? null
        : unwrap(
            await api.GET("/eval-sets/{eval_set_id}/versions/{version}/items", {
              params: {
                path: { eval_set_id: evalSetId, version: version.version },
                query: { limit: 200, ...(split ? { split } : {}) },
              },
            }),
          ),
    `${evalSetId}:${version?.version ?? ""}:${split}`,
  );

  const freeze = async () => {
    if (version === null) return;
    setBusy(true);
    setError(null);
    try {
      unwrap(
        await api.POST("/eval-sets/{eval_set_id}/versions/{version}/freeze", {
          params: {
            path: { eval_set_id: evalSetId, version: version.version },
          },
        }),
      );
      host.notify(`Version ${version.version} frozen.`, "success");
      versions.reload();
      detail.reload();
    } catch (failure) {
      setError(failure);
    } finally {
      setBusy(false);
    }
  };

  const set = detail.data;
  return (
    <div className="grid gap-4">
      <p className="text-sm">
        <Link to={{ name: "evalsets" }} className="text-link">
          Eval sets
        </Link>{" "}
        / {set?.name ?? "…"}
      </p>
      <ErrorNote error={detail.error} />
      <Section
        title={set ? `Eval set ${set.name}` : "Eval set"}
        description={set?.description || undefined}
      >
        {versions.data === undefined ? (
          <Loading />
        ) : (
          <DataTable<VersionOut>
            label="Versions"
            rows={versions.data.items}
            rowKey={(item) => item.id}
            selected={version?.id ?? null}
            onSelect={(item) => setChosen(item.version)}
            columns={[
              { header: "Version", cell: (item) => `v${item.version}` },
              { header: "Items", cell: (item) => item.item_count },
              { header: "Splits", cell: (item) => splitSummary(item) },
              {
                header: "State",
                cell: (item) => (
                  <StatusChip
                    status={item.frozen ? "frozen" : "draft"}
                    label={item.frozen ? "frozen" : "draft"}
                  />
                ),
              },
              {
                header: "Test split",
                cell: (item) =>
                  `${item.test_locked ? "locked" : "open"}, read ${item.test_reads}×`,
              },
              {
                header: "As of",
                cell: (item) => String(item.document.as_of ?? "–"),
              },
              {
                header: "Hash",
                cell: (item) => <code>{shortHash(item.document_hash)}</code>,
              },
            ]}
          />
        )}
      </Section>
      {version ? (
        <ReviewQueue
          evalSetId={evalSetId}
          onAdded={() => {
            setChosen(null);
            versions.reload();
            detail.reload();
          }}
        />
      ) : null}
      {version ? (
        <Section
          title={`Items in v${version.version}`}
          description="Gold evidence points at documents and evidence text, never at chunks."
          actions={
            <>
              <label className="flex items-center gap-1 text-xs font-medium">
                Split
                <select
                  className={inputClass}
                  value={split}
                  onChange={(event) =>
                    setSplit(event.target.value as "" | "dev" | "test")
                  }
                >
                  <option value="">all</option>
                  <option value="dev">dev</option>
                  <option value="test">test</option>
                </select>
              </label>
              {version.frozen ? null : (
                <Button
                  size="sm"
                  variant="secondary"
                  isDisabled={busy}
                  onPress={freeze}
                >
                  Freeze version
                </Button>
              )}
            </>
          }
        >
          <ErrorNote error={error ?? items.error} />
          {items.data === undefined || items.data === null ? (
            <Loading />
          ) : (
            <DataTable<ItemOut>
              label="Items"
              rows={items.data.items}
              rowKey={(item) => item.id}
              columns={[
                { header: "ID", cell: (item) => <code>{item.id}</code> },
                {
                  header: "Query",
                  cell: (item) => (
                    <span className="whitespace-pre-wrap">{item.query}</span>
                  ),
                },
                {
                  header: "Asker",
                  cell: (item) => String(item.asker.name ?? "–"),
                },
                {
                  header: "Evidence",
                  cell: (item) =>
                    item.expected.length
                      ? item.expected
                          .map(
                            (ref) =>
                              `${String(ref.doc_key)}@${String(ref.version ?? "*")}`,
                          )
                          .join(", ")
                      : "none (abstain)",
                },
                { header: "Tags", cell: (item) => item.tags.join(", ") },
                {
                  header: "Split",
                  cell: (item) => <StatusChip status={item.split} />,
                },
              ]}
            />
          )}
        </Section>
      ) : null}
    </div>
  );
}
