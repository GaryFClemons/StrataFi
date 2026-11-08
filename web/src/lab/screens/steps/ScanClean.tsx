// The Scan + Clean step. Configure: the
// scan checks, the policy (outcome → continue, flag or exclude; infected is
// always excluded), and the cleaning rules, which can be added, removed and
// reordered. Results: the outcome summary, each document's checks with
// reasons and finding counts (never values), the cleaning table, and a
// before/after diff of the cleaning on a chosen document, rebuilt by the API
// from the edit log with redacted values masked.

import { Button } from "@heroui/react";
import { useState } from "react";
import type { components } from "../../api";
import { unwrap, useLoad } from "../../api/load";
import type { RunOut } from "../../components/live";
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
import { formatFindings, tally } from "../../inspect";
import {
  DECISIONS,
  POLICY_OUTCOMES,
  appendItem,
  getAt,
  moveItem,
  removeItem,
  setAt,
  type SpecDocument,
} from "../../spec/editing";
import { defaultsFromSchema, type JsonSchema } from "../../spec/schemaForm";
import {
  Tasks,
  TaskStatus,
  summaryOf,
  useArtifactJson,
  value,
  type Summary,
  type TaskOut,
} from "./shared";

type ComponentOut = components["schemas"]["ComponentOut"];
type DiffOut = components["schemas"]["DiffOut"];

export function PolicyEditor({
  document,
  onChange,
}: {
  document: SpecDocument;
  onChange: (document: SpecDocument) => void;
}) {
  return (
    <fieldset
      className="grid gap-2 rounded-md border border-separator p-3"
      aria-label="Policy"
    >
      <legend className="px-1 text-xs font-semibold">Policy</legend>
      <p className="text-xs text-muted">
        What each scan outcome does to a document.
      </p>
      {POLICY_OUTCOMES.map((outcome) => {
        const path = `stages.scan.on_outcome.${outcome}`;
        const current = String(getAt(document, path) ?? "exclude");
        return (
          <label
            key={outcome}
            className="flex items-center justify-between gap-2 text-xs"
          >
            <span>{outcome.replace(/_/g, " ")}</span>
            <select
              className={inputClass}
              aria-label={`Policy for ${outcome.replace(/_/g, " ")}`}
              value={current}
              disabled={outcome === "infected"}
              onChange={(event) =>
                onChange(setAt(document, path, event.target.value))
              }
            >
              {DECISIONS.map((decision) => (
                <option key={decision} value={decision}>
                  {decision}
                </option>
              ))}
            </select>
          </label>
        );
      })}
    </fieldset>
  );
}

export function RuleList({
  document,
  components,
  onChange,
}: {
  document: SpecDocument;
  components: ComponentOut[];
  onChange: (document: SpecDocument) => void;
}) {
  const rules = (getAt(document, "stages.clean") ?? []) as {
    component: string;
  }[];
  const available = components.filter(
    (component) =>
      component.stage === "clean" &&
      !rules.some((rule) => rule.component === component.id),
  );
  const [adding, setAdding] = useState("");
  const chosen = adding || available[0]?.id || "";
  return (
    <div className="grid gap-2" aria-label="Cleaning rules">
      <p className="text-xs font-semibold">Cleaning rules, in order</p>
      {rules.length === 0 ? <Empty>No cleaning rules.</Empty> : null}
      {rules.map((rule, index) => (
        <SlotEditor
          key={`${rule.component}:${index}`}
          slot={{
            path: `stages.clean.${index}`,
            stage: "clean",
            label: `Rule ${index + 1}`,
          }}
          document={document}
          components={components}
          onChange={onChange}
          actions={
            <>
              <Button
                size="sm"
                variant="tertiary"
                aria-label={`Move rule ${index + 1} up`}
                isDisabled={index === 0}
                onPress={() =>
                  onChange(moveItem(document, "stages.clean", index, index - 1))
                }
              >
                ↑
              </Button>
              <Button
                size="sm"
                variant="tertiary"
                aria-label={`Move rule ${index + 1} down`}
                isDisabled={index === rules.length - 1}
                onPress={() =>
                  onChange(moveItem(document, "stages.clean", index, index + 1))
                }
              >
                ↓
              </Button>
              <Button
                size="sm"
                variant="tertiary"
                aria-label={`Remove rule ${index + 1}`}
                onPress={() =>
                  onChange(removeItem(document, "stages.clean", index))
                }
              >
                Remove
              </Button>
            </>
          }
        />
      ))}
      {available.length > 0 ? (
        <div className="flex flex-wrap items-end gap-2">
          <label className="grid gap-1 text-xs font-medium">
            Add a rule
            <select
              className={inputClass}
              value={chosen}
              onChange={(event) => setAdding(event.target.value)}
            >
              {available.map((component) => (
                <option key={component.id} value={component.id}>
                  {component.title}
                </option>
              ))}
            </select>
          </label>
          <Button
            size="sm"
            variant="secondary"
            onPress={() => {
              const component = available.find((item) => item.id === chosen);
              if (!component) return;
              onChange(
                appendItem(document, "stages.clean", {
                  component: component.id,
                  params: defaultsFromSchema(
                    component.params_schema as JsonSchema,
                  ),
                }),
              );
              setAdding("");
            }}
          >
            Add rule
          </Button>
        </div>
      ) : null}
    </div>
  );
}

interface ScanReport {
  outcome: string;
  decision: string;
  checks: {
    component: string;
    outcome: string;
    reason?: string | null;
    findings?: Record<string, number>;
  }[];
}

function ScanChecks({ cacheKey }: { cacheKey: string }) {
  const report = useArtifactJson<ScanReport>(cacheKey);
  if (report.error) return <ErrorNote error={report.error} />;
  if (!report.data) return <Loading />;
  return (
    <DataTable<ScanReport["checks"][number]>
      label="Checks"
      rows={report.data.checks}
      rowKey={(check) => check.component}
      columns={[
        { header: "Check", cell: (check) => check.component },
        {
          header: "Outcome",
          cell: (check) => <StatusChip status={check.outcome} />,
        },
        { header: "Reason", cell: (check) => check.reason ?? "" },
        { header: "Findings", cell: (check) => formatFindings(check.findings) },
      ]}
    />
  );
}

/** A cleaned document's edits, rule by rule: before and after with context. */
export function CleanDiff({ cacheKey }: { cacheKey: string }) {
  const { api } = useLab();
  const diff = useLoad<DiffOut>(
    async () =>
      unwrap(
        await api.GET("/artifacts/{cache_key}/diff", {
          params: { path: { cache_key: cacheKey }, query: { limit: 300 } },
        }),
      ),
    `diff:${cacheKey}`,
  );
  const [rule, setRule] = useState<string>("");
  if (diff.error) return <ErrorNote error={diff.error} />;
  if (!diff.data) return <Loading />;
  const data = diff.data;
  const hunks = rule
    ? data.hunks.filter((hunk) => hunk.rule === rule)
    : data.hunks;
  return (
    <section aria-label="Cleaning diff" className="grid gap-2">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span>
          {data.total_edits} edits · {data.before_chars} → {data.after_chars}{" "}
          characters
          {data.language ? ` · language ${data.language}` : ""}
        </span>
        {data.warnings.map((warning) => (
          <StatusChip key={warning} status="flag" label={warning} />
        ))}
        <label className="flex items-center gap-1">
          Rule
          <select
            className={inputClass}
            value={rule}
            onChange={(event) => setRule(event.target.value)}
          >
            <option value="">All ({data.total_edits})</option>
            {data.rules.map((item) => (
              <option key={item.rule} value={item.rule}>
                {item.rule} ({item.edits})
              </option>
            ))}
          </select>
        </label>
      </div>
      {hunks.length === 0 ? (
        <Empty>No edits: cleaning left this document as it was.</Empty>
      ) : (
        <ol className="grid gap-2">
          {hunks.map((hunk, index) => (
            <li
              key={`${hunk.rule}:${hunk.start}:${index}`}
              className="grid gap-1 rounded-md border border-separator p-2"
              data-testid="diff-hunk"
            >
              <span className="flex flex-wrap items-center gap-2 text-xs text-muted">
                <code>{hunk.rule}</code> {hunk.kind}
                {hunk.redaction ? (
                  <StatusChip status="flag" label="redaction" />
                ) : null}
              </span>
              <div className="grid gap-1 md:grid-cols-2">
                <pre
                  aria-label="Before"
                  className="whitespace-pre-wrap break-words rounded bg-danger-soft p-1 font-mono text-xs"
                >
                  {hunk.before}
                </pre>
                <pre
                  aria-label="After"
                  className="whitespace-pre-wrap break-words rounded bg-success-soft p-1 font-mono text-xs"
                >
                  {hunk.after}
                </pre>
              </div>
            </li>
          ))}
        </ol>
      )}
      {data.truncated ? (
        <p className="text-xs text-muted">
          Showing the first {data.hunks.length} edits.
        </p>
      ) : null}
    </section>
  );
}

export function ScanInspector({ run }: { run: RunOut }) {
  const [chosenScan, setChosenScan] = useState<string | null>(null);
  const [chosenClean, setChosenClean] = useState<string | null>(null);
  return (
    <div className="grid gap-4">
      <Tasks run={run} stage="scan" empty="No documents scanned yet.">
        {(tasks) => {
          const summaries = tasks.map(summaryOf);
          return (
            <div className="grid gap-2">
              <div className="flex flex-wrap gap-2" aria-label="Outcomes">
                {tally(summaries, "outcome").map(([outcome, count]) => (
                  <StatusChip
                    key={outcome}
                    status={outcome}
                    label={`${outcome}: ${count}`}
                  />
                ))}
              </div>
              <DataTable<TaskOut>
                label="Scan outcomes"
                rows={tasks}
                rowKey={(task) => task.cache_key}
                selected={chosenScan}
                onSelect={(task) => setChosenScan(task.cache_key)}
                columns={[
                  { header: "Document", cell: (task) => task.entry ?? "–" },
                  {
                    header: "Outcome",
                    cell: (task) => (
                      <StatusChip status={value(summaryOf(task), "outcome")} />
                    ),
                  },
                  {
                    header: "Decision",
                    cell: (task) => (
                      <StatusChip status={value(summaryOf(task), "decision")} />
                    ),
                  },
                  {
                    header: "Findings",
                    cell: (task) =>
                      formatFindings(summaryOf(task).findings as Summary),
                  },
                  {
                    header: "Status",
                    cell: (task) => <TaskStatus task={task} />,
                  },
                ]}
              />
              {chosenScan ? <ScanChecks cacheKey={chosenScan} /> : null}
            </div>
          );
        }}
      </Tasks>
      <Tasks run={run} stage="clean" empty="No documents cleaned yet.">
        {(tasks) => {
          const done = tasks.filter(
            (task) => summaryOf(task).edits !== undefined,
          );
          const shown = chosenClean ?? done[0]?.cache_key ?? null;
          return (
            <div className="grid gap-2">
              <DataTable<TaskOut>
                label="Cleaning"
                rows={tasks}
                rowKey={(task) => task.cache_key}
                selected={shown}
                onSelect={(task) => setChosenClean(task.cache_key)}
                columns={[
                  { header: "Document", cell: (task) => task.entry ?? "–" },
                  {
                    header: "Edits",
                    cell: (task) => value(summaryOf(task), "edits"),
                  },
                  {
                    header: "Redactions",
                    cell: (task) => value(summaryOf(task), "redactions"),
                  },
                  {
                    header: "Language",
                    cell: (task) => value(summaryOf(task), "language"),
                  },
                  {
                    header: "Characters",
                    cell: (task) => value(summaryOf(task), "chars"),
                  },
                  {
                    header: "Status",
                    cell: (task) =>
                      summaryOf(task).excluded ? (
                        <StatusChip
                          status="exclude"
                          label={String(summaryOf(task).excluded)}
                        />
                      ) : (
                        <TaskStatus task={task} />
                      ),
                  },
                ]}
              />
              {shown ? <CleanDiff key={shown} cacheKey={shown} /> : null}
            </div>
          );
        }}
      </Tasks>
    </div>
  );
}
