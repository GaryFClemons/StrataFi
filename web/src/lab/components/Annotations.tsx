// A run's title, tags, Markdown note and archive. The note renders as
// text in Markdown's shape: nothing becomes HTML, and links stay text.

import { Button } from "@heroui/react";
import { useState, type FormEvent, type ReactNode } from "react";
import type { components } from "../api";
import { unwrap, useLoad } from "../api/load";
import { useLab } from "../context";
import { parseMarkdown, type Inline } from "../markdown";
import { formatTags, parseTags } from "../tracking";
import { ErrorNote, Loading, Section, inputClass } from "./ui";

type AnnotationOut = components["schemas"]["AnnotationOut"];
export type AnnotatedKind = "run" | "eval_run" | "experiment";

function inline(parts: Inline[]): ReactNode[] {
  return parts.map((part, index) => {
    if (part.kind === "strong") return <strong key={index}>{part.text}</strong>;
    if (part.kind === "em") return <em key={index}>{part.text}</em>;
    if (part.kind === "code") {
      return (
        <code key={index} className="rounded bg-default px-1 text-xs">
          {part.text}
        </code>
      );
    }
    return <span key={index}>{part.text}</span>;
  });
}

/** A Markdown note, drawn as elements from parsed blocks; never as HTML. */
export function Note({ text }: { text: string }) {
  const blocks = parseMarkdown(text);
  return (
    <div className="grid gap-2 text-sm" data-testid="note">
      {blocks.map((block, index) => {
        switch (block.kind) {
          case "heading":
            return (
              <p key={index} className="font-semibold">
                {inline(block.inline)}
              </p>
            );
          case "paragraph":
            return <p key={index}>{inline(block.inline)}</p>;
          case "code":
            return (
              <pre
                key={index}
                className="overflow-auto rounded-md border border-separator p-2 font-mono text-xs"
              >
                {block.text}
              </pre>
            );
          case "list": {
            const items = block.items.map((item, at) => (
              <li key={at}>{inline(item)}</li>
            ));
            return block.ordered ? (
              <ol key={index} className="list-decimal pl-5">
                {items}
              </ol>
            ) : (
              <ul key={index} className="list-disc pl-5">
                {items}
              </ul>
            );
          }
        }
        return null;
      })}
    </div>
  );
}

async function load(
  api: ReturnType<typeof useLab>["api"],
  kind: AnnotatedKind,
  id: string,
): Promise<AnnotationOut> {
  if (kind === "run") {
    return unwrap(
      await api.GET("/runs/{run_id}/annotation", {
        params: { path: { run_id: id } },
      }),
    );
  }
  if (kind === "experiment") {
    return unwrap(
      await api.GET("/experiments/{experiment_id}/annotation", {
        params: { path: { experiment_id: id } },
      }),
    );
  }
  return unwrap(
    await api.GET("/eval-runs/{eval_run_id}/annotation", {
      params: { path: { eval_run_id: id } },
    }),
  );
}

type AnnotationIn = components["schemas"]["AnnotationIn"];

async function save(
  api: ReturnType<typeof useLab>["api"],
  kind: AnnotatedKind,
  id: string,
  body: AnnotationIn,
): Promise<AnnotationOut> {
  if (kind === "run") {
    return unwrap(
      await api.PATCH("/runs/{run_id}/annotation", {
        params: { path: { run_id: id } },
        body,
      }),
    );
  }
  if (kind === "experiment") {
    return unwrap(
      await api.PATCH("/experiments/{experiment_id}/annotation", {
        params: { path: { experiment_id: id } },
        body,
      }),
    );
  }
  return unwrap(
    await api.PATCH("/eval-runs/{eval_run_id}/annotation", {
      params: { path: { eval_run_id: id } },
      body,
    }),
  );
}

/** Edit a run's title, tags and note, and archive or restore it. */
export function Annotations({
  kind,
  id,
  onChange,
}: {
  kind: AnnotatedKind;
  id: string;
  onChange?: () => void;
}) {
  const { api, host } = useLab();
  const loaded = useLoad(() => load(api, kind, id), `${kind}:${id}`);
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState("");
  const [tags, setTags] = useState("");
  const [note, setNote] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const current = loaded.data;

  const begin = () => {
    setTitle(current?.title ?? "");
    setTags(formatTags(current?.tags));
    setNote(current?.note ?? "");
    setEditing(true);
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const parsed = parseTags(tags);
    if (parsed.invalid.length) {
      host.notify("Tags are key=value, separated by commas.", "warning");
      return;
    }
    const removed = Object.keys(current?.tags ?? {}).filter(
      (key) => !(key in parsed.tags),
    );
    setBusy(true);
    setError(null);
    try {
      await save(api, kind, id, {
        title,
        note,
        set_tags: parsed.tags,
        delete_tags: removed,
      });
      setEditing(false);
      loaded.reload();
      onChange?.();
    } catch (failure) {
      setError(failure);
    } finally {
      setBusy(false);
    }
  };

  const archive = async (archived: boolean) => {
    setError(null);
    try {
      await save(api, kind, id, { archived });
      loaded.reload();
      onChange?.();
      host.notify(
        archived ? "Archived: hidden from lists and searches." : "Restored.",
        "success",
      );
    } catch (failure) {
      setError(failure);
    }
  };

  return (
    <Section
      title="Title, tags and note"
      description="Yours to name, group and explain runs with. Never logged."
      actions={
        current && !editing ? (
          <>
            <Button size="sm" variant="secondary" onPress={begin}>
              Edit
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onPress={() => archive(!current.archived)}
            >
              {current.archived ? "Restore" : "Archive"}
            </Button>
          </>
        ) : null
      }
    >
      {current === undefined ? (
        loaded.error ? (
          <ErrorNote error={loaded.error} />
        ) : (
          <Loading />
        )
      ) : editing ? (
        <form onSubmit={submit} aria-label="Annotate" className="grid gap-2">
          <label className="grid gap-1 text-xs font-medium">
            Title
            <input
              className={inputClass}
              value={title}
              maxLength={120}
              onChange={(event) => setTitle(event.target.value)}
            />
          </label>
          <label className="grid gap-1 text-xs font-medium">
            Tags (key=value, separated by commas)
            <input
              className={inputClass}
              value={tags}
              onChange={(event) => setTags(event.target.value)}
            />
          </label>
          <label className="grid gap-1 text-xs font-medium">
            Note (Markdown)
            <textarea
              className={`${inputClass} min-h-28 font-mono`}
              value={note}
              maxLength={20000}
              onChange={(event) => setNote(event.target.value)}
            />
          </label>
          <div className="flex gap-2">
            <Button type="submit" size="sm" isDisabled={busy}>
              Save
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onPress={() => setEditing(false)}
            >
              Cancel
            </Button>
          </div>
        </form>
      ) : (
        <div className="grid gap-2" data-testid="annotation">
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="font-semibold">{current.title ?? "No title"}</span>
            {current.archived ? (
              <span className="rounded bg-default px-2 text-xs">archived</span>
            ) : null}
            {Object.entries(current.tags).map(([key, value]) => (
              <span
                key={key}
                className="rounded-full border border-separator px-2 text-xs"
                data-testid="tag"
              >
                {key}={value}
              </span>
            ))}
          </div>
          {current.note ? <Note text={current.note} /> : null}
        </div>
      )}
      <ErrorNote error={error} />
    </Section>
  );
}
