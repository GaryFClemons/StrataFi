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
} from "../components/ui";
import { Link, useLab, useNavigate } from "../context";
import { formatEstimate, type Estimate } from "../evals";
import { configRows, downloadText } from "../profiles";
import { formatTime, shortHash } from "../format";

type ProfileOut = components["schemas"]["ProfileOut"];
type Row = Record<string, unknown>;

function estimate(value: unknown): Estimate | null {
  return value !== null && typeof value === "object"
    ? (value as Estimate)
    : null;
}

/** Knowledge profiles, newest version first, and an import. */
export function Profiles() {
  const { api, host } = useLab();
  const navigate = useNavigate();
  const profiles = useLoad(
    async () =>
      unwrap(await api.GET("/profiles", { params: { query: { limit: 100 } } })),
    "profiles",
  );
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<unknown>(null);

  const upload = async (event: FormEvent) => {
    event.preventDefault();
    if (file === null) return;
    setError(null);
    try {
      const imported = unwrap(
        await api.POST("/profiles/import", {
          body: { content: await file.text() },
        }),
      );
      host.notify(
        imported.created === false
          ? `Unchanged: ${imported.name} v${imported.version} holds this profile.`
          : `Imported ${imported.name} v${imported.version}.`,
        "success",
      );
      navigate({ name: "profile", profileId: imported.id });
    } catch (failure) {
      setError(failure);
    }
  };

  return (
    <div className="grid gap-4">
      <Section
        title="Knowledge profiles"
        description="Winning configurations, the pipeline and the retrieval config, with their evidence: exported from trials confirmed on the test split, and reviewed by an owner."
      >
        {profiles.data === undefined ? (
          profiles.error ? (
            <ErrorNote error={profiles.error} />
          ) : (
            <Loading />
          )
        ) : profiles.data.items.length === 0 ? (
          <Empty>
            No profiles yet. Confirm a trial on the test split, then export it
            from its experiment.
          </Empty>
        ) : (
          <DataTable<ProfileOut>
            label="Knowledge profiles"
            rows={profiles.data.items}
            rowKey={(profile) => profile.id}
            columns={[
              {
                header: "Profile",
                cell: (profile) => (
                  <Link
                    to={{ name: "profile", profileId: profile.id }}
                    className="text-link"
                  >
                    {profile.name} v{profile.version}
                  </Link>
                ),
              },
              {
                header: "Review",
                cell: (profile) => <StatusChip status={profile.status} />,
              },
              { header: "Source", cell: (profile) => profile.source },
              {
                header: "Created",
                cell: (profile) => formatTime(profile.created_at),
              },
            ]}
          />
        )}
        <form
          onSubmit={upload}
          aria-label="Import a profile"
          className="mt-3 flex flex-wrap items-end gap-2"
        >
          <label className="grid gap-1 text-xs font-medium">
            Profile file (stratafi.knowledge-profile.v1, YAML or JSON)
            <input
              type="file"
              accept=".yaml,.yml,.json"
              className="text-sm"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              required
            />
          </label>
          <Button type="submit" size="sm" variant="secondary">
            Import
          </Button>
        </form>
        <ErrorNote error={error} />
      </Section>
    </div>
  );
}

/** One profile version: configuration, evidence, review, diff and export. */
export function ProfileDetail({ profileId }: { profileId: string }) {
  const { api, host } = useLab();
  const [reload, setReload] = useState(0);
  const loaded = useLoad(
    async () =>
      unwrap(
        await api.GET("/profiles/{profile_id}", {
          params: { path: { profile_id: profileId } },
        }),
      ),
    `${profileId}:${reload}`,
  );
  const diff = useLoad(
    async () =>
      unwrap(
        await api.GET("/profiles/{profile_id}/diff", {
          params: { path: { profile_id: profileId } },
        }),
      ),
    profileId,
  );
  const [error, setError] = useState<unknown>(null);
  const profile = loaded.data;
  if (profile === undefined) {
    return loaded.error ? <ErrorNote error={loaded.error} /> : <Loading />;
  }
  const document = profile.document as Row;
  const pipeline = (document.pipeline ?? {}) as Row;
  const retrieval = (document.retrieval ?? {}) as Row;
  const evidence = (document.evidence ?? {}) as Row;
  const experiment = (evidence.experiment ?? null) as Row | null;
  const trial = (evidence.trial ?? null) as Row | null;
  const objective = (evidence.objective ?? {}) as Row;
  const metric = String(objective.metric ?? "objective");

  const decide = async (decision: "approved" | "rejected") => {
    setError(null);
    try {
      unwrap(
        await api.POST("/profiles/{profile_id}/approvals", {
          params: { path: { profile_id: profileId } },
          body: { decision },
        }),
      );
      host.notify(`The profile is ${decision}.`, "success");
      setReload((value) => value + 1);
    } catch (failure) {
      setError(failure);
    }
  };

  const download = async (format: "yaml" | "json") => {
    setError(null);
    try {
      const text = unwrap(
        await api.GET("/profiles/{profile_id}/export", {
          params: { path: { profile_id: profileId }, query: { format } },
          parseAs: "text",
        }),
      ) as unknown as string;
      downloadText(
        text,
        `${profile.name}-v${profile.version}.${format}`,
        format === "yaml" ? "application/yaml" : "application/json",
      );
    } catch (failure) {
      setError(failure);
    }
  };

  return (
    <div className="grid gap-4">
      <p className="text-sm">
        <Link to={{ name: "profiles" }} className="text-link">
          Profiles
        </Link>{" "}
        / {profile.name} v{profile.version}
      </p>
      <Section
        title={`${profile.name} v${profile.version}`}
        description={`${String(document.description ?? "")} Hash ${shortHash(profile.document_hash)}.`.trim()}
        actions={<StatusChip status={profile.status} />}
      >
        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            variant="secondary"
            onPress={() => void download("yaml")}
          >
            Download YAML
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onPress={() => void download("json")}
          >
            Download JSON
          </Button>
        </div>
        <ErrorNote error={error} />
      </Section>
      <Section
        title="Configuration"
        description={`The winning pipeline, ${String(pipeline.name ?? "–")}, and retrieval config, ${String(retrieval.name ?? "–")}, resolved with every default.`}
      >
        <div className="grid gap-3 lg:grid-cols-2">
          {(
            [
              ["Pipeline", pipeline],
              ["Retrieval", retrieval],
            ] as const
          ).map(([label, part]) => (
            <DataTable
              key={label}
              label={label}
              rows={configRows(part)}
              rowKey={(row) => row.path}
              columns={[
                { header: label, cell: (row) => <code>{row.path}</code> },
                { header: "Value", cell: (row) => row.value },
              ]}
            />
          ))}
        </div>
      </Section>
      <Section
        title="Evidence"
        description="Chosen on dev, confirmed once on the locked test split."
      >
        <div className="grid gap-1 text-sm" data-testid="profile-evidence">
          {experiment ? (
            <span>
              Experiment{" "}
              <Link
                to={{ name: "experiment", experimentId: String(experiment.id) }}
                className="text-link"
              >
                {String(experiment.name)}
              </Link>
              , trial {String(trial?.ordinal ?? "–")}
            </span>
          ) : (
            <span>Imported.</span>
          )}
          <span>
            {metric} on dev: {formatEstimate(estimate(evidence.dev))}
          </span>
          <span>
            {metric} on test: {formatEstimate(estimate(evidence.test))}
          </span>
        </div>
      </Section>
      <Section
        title="Review"
        description="An owner approves or rejects each version; its author's own decision is marked a self-review."
        actions={
          <>
            <Button size="sm" onPress={() => void decide("approved")}>
              Approve
            </Button>
            <Button
              size="sm"
              variant="danger"
              onPress={() => void decide("rejected")}
            >
              Reject
            </Button>
          </>
        }
      >
        {profile.approvals.length === 0 ? (
          <Empty>Not reviewed yet.</Empty>
        ) : (
          <DataTable
            label="Review decisions"
            rows={profile.approvals}
            rowKey={(row) => `${row.decided_by}:${row.decided_at}`}
            columns={[
              {
                header: "Decision",
                cell: (row) => <StatusChip status={row.decision} />,
              },
              {
                header: "By",
                cell: (row) =>
                  row.self_review
                    ? `${row.decided_by} (self-review)`
                    : row.decided_by,
              },
              { header: "When", cell: (row) => formatTime(row.decided_at) },
            ]}
          />
        )}
      </Section>
      <Section
        title="Changes"
        description="Against the previous version of this profile."
      >
        {diff.data === undefined ? (
          <Loading />
        ) : diff.data.against === null ? (
          <Empty>The first version.</Empty>
        ) : diff.data.changes.length === 0 ? (
          <Empty>The same configuration as the previous version.</Empty>
        ) : (
          <DataTable
            label="Changes against the previous version"
            rows={diff.data.changes}
            rowKey={(row) => row.path}
            columns={[
              { header: "Path", cell: (row) => <code>{row.path}</code> },
              { header: "Change", cell: (row) => row.op },
              { header: "Before", cell: (row) => JSON.stringify(row.before) },
              { header: "After", cell: (row) => JSON.stringify(row.after) },
            ]}
          />
        )}
      </Section>
    </div>
  );
}
