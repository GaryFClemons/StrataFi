import { unwrap, useLoad } from "../api/load";
import type { RunOut } from "../components/live";
import {
  DataTable,
  Empty,
  ErrorNote,
  Loading,
  Section,
  StatusChip,
} from "../components/ui";
import { Link, useLab } from "../context";
import { formatTime, progress, shortHash } from "../format";

export function Home() {
  const { api } = useLab();
  const runs = useLoad(
    async () =>
      unwrap(await api.GET("/runs", { params: { query: { limit: 5 } } })).items,
    "home-runs",
  );
  return (
    <div className="grid gap-4">
      <Section
        title="Retrieval lab"
        description="Run every RAG ETL step, track every run, and export the configuration that wins."
      >
        <ol className="grid list-decimal gap-1 pl-5 text-sm">
          <li>
            <Link to={{ name: "documents" }} className="text-link">
              Upload documents
            </Link>{" "}
            into a corpus and freeze a version.
          </li>
          <li>
            <Link
              to={{ name: "pipelines", step: "extract" }}
              className="text-link"
            >
              Configure the five steps
            </Link>{" "}
            and run to any of them on a sample.
          </li>
          <li>
            <Link to={{ name: "search" }} className="text-link">
              Search
            </Link>{" "}
            what a run indexed.
          </li>
        </ol>
      </Section>
      <Section title="Recent runs">
        <ErrorNote error={runs.error} />
        {runs.data === undefined ? (
          <Loading />
        ) : runs.data.length === 0 ? (
          <Empty>No runs yet.</Empty>
        ) : (
          <DataTable<RunOut>
            label="Recent runs"
            rows={runs.data}
            rowKey={(run) => run.id}
            columns={[
              {
                header: "Run",
                cell: (run) => (
                  <Link
                    to={{ name: "run", runId: run.id }}
                    className="text-link"
                  >
                    <code>{shortHash(run.id, 8)}</code>
                  </Link>
                ),
              },
              {
                header: "Status",
                cell: (run) => <StatusChip status={run.status} />,
              },
              { header: "Until", cell: (run) => run.until ?? "index" },
              {
                header: "Tasks",
                cell: (run) => {
                  const { finished, total } = progress(run.counts.tasks);
                  return `${finished}/${total}`;
                },
              },
              { header: "Started", cell: (run) => formatTime(run.created_at) },
            ]}
          />
        )}
      </Section>
    </div>
  );
}
