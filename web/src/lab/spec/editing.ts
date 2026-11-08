// Editing a pipeline spec from the step screens. Forms, the YAML view and the
// SDK edit the same document (one document per configuration), so
// these are pure functions over that document: each returns a new one.

import type { StepId } from "../routes";

export type SpecDocument = Record<string, unknown>;

export const STAGES = [
  "extract",
  "scan",
  "parse",
  "clean",
  "chunk",
  "enrich",
  "embed",
  "index",
] as const;
export type Stage = (typeof STAGES)[number];

export interface StepDefinition {
  id: StepId;
  title: string;
  /** The run's last stage for "Run to this step"; null when the step doesn't run. */
  until: Stage | null;
  /** The stages whose tasks the step's inspector shows. */
  stages: Stage[];
}

// Clean comes after parse in the pipeline but shares the Scan + Clean screen,
// so running to it runs parse too.
export const STEPS: StepDefinition[] = [
  { id: "extract", title: "Extract", until: "extract", stages: ["extract"] },
  {
    id: "scan",
    title: "Scan + Clean",
    until: "clean",
    stages: ["scan", "clean"],
  },
  { id: "parse", title: "Parse", until: "parse", stages: ["parse"] },
  { id: "chunk", title: "Chunk", until: "enrich", stages: ["chunk", "enrich"] },
  { id: "embed", title: "Embed", until: "index", stages: ["embed", "index"] },
  { id: "search", title: "Search", until: null, stages: [] },
];

export function stepOf(id: StepId): StepDefinition {
  return STEPS.find((step) => step.id === id)!;
}

/** Whether a run that went to `until` (null: every stage) has reached `stage`. */
export function reaches(
  until: string | null | undefined,
  stage: Stage,
): boolean {
  if (until === null || until === undefined) return true;
  const last = STAGES.indexOf(until as Stage);
  return last >= 0 && STAGES.indexOf(stage) <= last;
}

/** One component in the spec that a step configures. */
export interface Slot {
  /** Where its {component, params} live, e.g. "stages.chunk". */
  path: string;
  /** The registry stage to pick components from. */
  stage: Stage;
  label: string;
  /** Only components whose ID starts with this. */
  prefix?: string;
  /** Its component can't be swapped here, only its parameters. */
  fixed?: boolean;
  /** Only these components fit it (a dense or a keyword index). */
  only?: string[];
}

export interface ComponentRef {
  component: string;
  params: Record<string, unknown>;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function keys(path: string): string[] {
  return path.split(".").filter(Boolean);
}

/** The value at a dotted path ("stages.clean.0"), or undefined. */
export function getAt(document: unknown, path: string): unknown {
  let current: unknown = document;
  for (const key of keys(path)) {
    if (Array.isArray(current)) current = current[Number(key)];
    else if (isObject(current)) current = current[key];
    else return undefined;
  }
  return current;
}

/** A copy of the document with the value at `path` replaced (or removed, for undefined). */
export function setAt(
  document: SpecDocument,
  path: string,
  value: unknown,
): SpecDocument {
  const copy = structuredClone(document);
  const parts = keys(path);
  let current: unknown = copy;
  for (const [index, key] of parts.entries()) {
    const last = index === parts.length - 1;
    const container = current as Record<string, unknown> | unknown[];
    const slot = Array.isArray(container) ? Number(key) : key;
    if (last) {
      if (value === undefined) {
        if (Array.isArray(container)) container.splice(slot as number, 1);
        else delete container[slot as string];
      } else {
        (container as Record<string | number, unknown>)[slot] = value;
      }
      break;
    }
    let next = (container as Record<string | number, unknown>)[slot];
    if (!isObject(next) && !Array.isArray(next)) {
      next = /^\d+$/.test(parts[index + 1]!) ? [] : {};
      (container as Record<string | number, unknown>)[slot] = next;
    }
    current = next;
  }
  return copy;
}

/** The component reference at `path`, if there is one. */
export function refAt(document: unknown, path: string): ComponentRef | null {
  const value = getAt(document, path);
  if (!isObject(value) || typeof value.component !== "string") return null;
  return {
    component: value.component,
    params: isObject(value.params) ? value.params : {},
  };
}

/** Swap the component at `path`; its parameters start from the new one's defaults. */
export function withComponent(
  document: SpecDocument,
  path: string,
  component: string,
  defaults: Record<string, unknown>,
): SpecDocument {
  const current = getAt(document, path);
  const rest = isObject(current) ? current : {};
  return setAt(document, path, {
    ...rest,
    component,
    params: { ...defaults },
  });
}

/** Set one parameter of the component at `path`. */
export function withParam(
  document: SpecDocument,
  path: string,
  name: string,
  value: unknown,
): SpecDocument {
  if (refAt(document, path) === null) {
    throw new Error(`no component at ${path}`);
  }
  return setAt(document, `${path}.params.${name}`, value);
}

function listSlots(
  document: unknown,
  path: string,
  stage: Stage,
  label: string,
  prefix?: string,
): Slot[] {
  const items = getAt(document, path);
  if (!Array.isArray(items)) return [];
  return items.map((_, index) => ({
    path: `${path}.${index}`,
    stage,
    label: `${label} ${index + 1}`,
    prefix,
  }));
}

/** The components a step configures, as the document has them. */
export function slotsFor(step: StepId, document: unknown): Slot[] {
  switch (step) {
    case "extract":
      return [
        {
          path: "stages.extract.component",
          stage: "extract",
          label: "Extractor",
          prefix: "extract/",
        },
        ...listSlots(
          document,
          "stages.extract.unpack",
          "extract",
          "Unpacker",
          "unpack/",
        ),
      ];
    case "scan":
      // Cleaning rules have their own list editor (add, remove, reorder).
      return listSlots(document, "stages.scan.checks", "scan", "Check");
    case "parse":
      // The routing table has its own editor (add and remove routes).
      return [];
    case "chunk":
      return [
        { path: "stages.chunk", stage: "chunk", label: "Chunker" },
        { path: "stages.enrich", stage: "enrich", label: "Enrichment" },
      ];
    case "embed":
      return [
        { path: "stages.embed", stage: "embed", label: "Embedder" },
        ...(["dense", "keyword"] as const)
          .filter((kind) => refAt(document, `stages.index.${kind}`) !== null)
          .map((kind) => ({
            path: `stages.index.${kind}`,
            stage: "index" as const,
            label: kind === "dense" ? "Dense index" : "Keyword index",
            // Dense: exact or pgvector HNSW; keyword: BM25 or Postgres full-text.
            only:
              kind === "dense"
                ? ["index/exact", "index/pgvector-hnsw"]
                : ["index/bm25", "index/postgres-fts"],
          })),
      ];
    case "search":
      return [];
  }
}

/** JSON with sorted keys: two documents are the same configuration when these match. */
export function stableStringify(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
  if (isObject(value)) {
    const entries = Object.keys(value)
      .sort()
      .filter((key) => value[key] !== undefined)
      .map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`);
    return `{${entries.join(",")}}`;
  }
  return JSON.stringify(value);
}

export function sameDocument(a: unknown, b: unknown): boolean {
  return stableStringify(a) === stableStringify(b);
}

/** The sizes "Run to this step" offers; null runs every entry. */
export const SAMPLE_SIZES: (number | null)[] = [5, 20, null];

/** A run's sample for a size: a fixed seed, so the same size picks the same entries. */
export function sampleFor(
  size: number | null,
): { size: number; seed: number } | null {
  return size === null ? null : { size, seed: 0 };
}

/** A copy with the list at `path` reordered: the item at `from` moves to `to`. */
export function moveItem(
  document: SpecDocument,
  path: string,
  from: number,
  to: number,
): SpecDocument {
  const items = getAt(document, path);
  if (!Array.isArray(items)) throw new Error(`no list at ${path}`);
  if (to < 0 || to >= items.length || from === to) return document;
  const copy = [...items];
  const [item] = copy.splice(from, 1);
  copy.splice(to, 0, item);
  return setAt(document, path, copy);
}

/** A copy with the item at `index` of the list at `path` removed. */
export function removeItem(
  document: SpecDocument,
  path: string,
  index: number,
): SpecDocument {
  const items = getAt(document, path);
  if (!Array.isArray(items)) throw new Error(`no list at ${path}`);
  return setAt(
    document,
    path,
    items.filter((_, position) => position !== index),
  );
}

/** A copy with `item` appended to the list at `path` (made if missing). */
export function appendItem(
  document: SpecDocument,
  path: string,
  item: unknown,
): SpecDocument {
  const items = getAt(document, path);
  return setAt(document, path, [...(Array.isArray(items) ? items : []), item]);
}

/** The scan outcomes a policy maps; infected is always excluded (the security floor). */
export const POLICY_OUTCOMES = [
  "infected",
  "encrypted",
  "malformed",
  "limit_exceeded",
  "pii",
  "error",
  "duplicate",
] as const;
export const DECISIONS = ["continue", "flag", "exclude"] as const;
