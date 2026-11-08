// Pure helpers for the experiment screens: the
// plan's unique work and budget, factor order and trial labels. Numbers and
// names only.

export const ACTIVE_EXPERIMENT_STATUSES = ["running", "stopping", "cancelling"];

export function isExperimentActive(status: string): boolean {
  return ACTIVE_EXPERIMENT_STATUSES.includes(status);
}

/** The pipeline's stages, in order: plans list their work this way. */
export const STAGE_ORDER = [
  "extract",
  "scan",
  "parse",
  "clean",
  "chunk",
  "enrich",
  "embed",
  "index",
];

export interface StageWork {
  stage: string;
  tasks: number;
  unique: number;
  cached: number;
  to_run: number;
}

type Plan = Record<string, unknown> | null | undefined;

function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

function number(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

/** The plan's work per stage, in pipeline order. */
export function workRows(plan: Plan): StageWork[] {
  const stages = record(record(record(plan).work).stages);
  return STAGE_ORDER.filter((stage) => stage in stages).map((stage) => {
    const work = record(stages[stage]);
    return {
      stage,
      tasks: number(work.tasks),
      unique: number(work.unique),
      cached: number(work.cached),
      to_run: number(work.to_run),
    };
  });
}

/** How much of "every trial alone" sharing saves: 0 to 1. */
export function sharedShare(rows: StageWork[]): number {
  const tasks = rows.reduce((sum, row) => sum + row.tasks, 0);
  const unique = rows.reduce((sum, row) => sum + row.unique, 0);
  return tasks > 0 ? 1 - unique / tasks : 0;
}

export interface BudgetRow {
  name: string;
  label: string;
  planned: string;
  max: string;
  over: boolean;
}

const BUDGET_LABELS: Record<string, string> = {
  trials: "Trials",
  cost_usd: "Cost (USD)",
  hours: "Hours",
};

function show(value: unknown, digits = 2): string {
  if (value === null || value === undefined) return "no limit";
  return typeof value === "number"
    ? Number.isInteger(value)
      ? String(value)
      : value.toFixed(digits)
    : String(value);
}

/** The plan's budget check, one row per budget. */
export function budgetRows(plan: Plan): BudgetRow[] {
  const budget = record(record(plan).budget);
  return Object.keys(BUDGET_LABELS)
    .filter((name) => name in budget)
    .map((name) => {
      const check = record(budget[name]);
      const planned = check.planned ?? check.estimated;
      return {
        name,
        label: BUDGET_LABELS[name] ?? name,
        planned: show(planned, name === "hours" ? 3 : 2),
        max: show(check.max),
        over: check.over === true,
      };
    });
}

/** The factors' paths in the order the experiment lists them (the grid's). */
export function factorPaths(plan: Plan): string[] {
  const factors = record(plan).factors;
  if (!Array.isArray(factors)) return [];
  return factors
    .map((factor) => record(factor).path)
    .filter((path): path is string => typeof path === "string");
}

/** A trial's levels, in factor order: "chunk/fixed-tokens size=256 · embed-local". */
export function trialLabel(
  values: Record<string, string>,
  paths: string[],
): string {
  const ordered = paths.length > 0 ? paths : Object.keys(values).sort();
  return ordered
    .filter((path) => path in values)
    .map((path) => values[path])
    .join(" · ");
}

export interface HalvingRung {
  rung: number;
  queries: number;
  keep: number;
}

/** Successive halving's rungs from the plan; none for another strategy. */
export function halvingRungs(plan: Plan): HalvingRung[] {
  const rungs = record(record(plan).search).rungs;
  if (!Array.isArray(rungs)) return [];
  return rungs.map((rung, index) => ({
    rung: index + 1,
    queries: number(record(rung).queries),
    keep: number(record(rung).keep),
  }));
}

/** A trial's way through the rungs: "10 → 30 → 63", or "10 (pruned)". */
export function rungTrail(rungs: readonly Record<string, unknown>[]): string {
  if (rungs.length === 0) return "–";
  const sizes = rungs.map((rung) => String(rung.queries ?? "?")).join(" → ");
  const last = rungs[rungs.length - 1];
  if (last.kept === false) return `${sizes} (pruned)`;
  if (last.kept === null) return `${sizes} (stopped)`;
  return sizes;
}

/** A start for a new experiment's YAML: names to replace with the workspace's own. */
export const EXAMPLE_EXPERIMENT = `schema: stratafi.experiment.v1
name: chunking-and-models
corpus: halcyon@latest
eval_set: {name: halcyon-retrieval}
base: {pipeline: baseline-v1, retrieval: hybrid-v1}
objective: {metric: recall@5, checkpoint: final, goal: maximize}
constraints:
  - {metric: must_not_return_violations, max: 0}
  - {metric: latency_ms.p95, max: 800}
factors:
  chunk.params.max_tokens: [400, 800]
  rerank: [rerank/none, {component: rerank/gateway}]
strategy: grid
budget: {max_trials: 20, max_cost_usd: 5}
`;

export interface Routing {
  path: string;
  best: Record<
    string,
    {
      level: number;
      label: string;
      value: unknown;
      mean: number;
      trials: number;
    }
  >;
}

/** The media types a routing table can name: no "unknown". */
function routable(route: Routing): [string, unknown][] {
  return Object.entries(route.best)
    .filter(([type]) => type.includes("/"))
    .map(([type, best]) => [type, best.value ?? best.label]);
}

/** A routing suggestion as the parse stage's YAML: one route per gold file type. */
export function routingYaml(route: Routing): string {
  const lines = [
    `# The best ${route.path} per gold file type`,
    "parse:",
    "  route:",
  ];
  for (const [type, value] of routable(route)) {
    lines.push(
      `    ${type}: ${typeof value === "string" ? value : JSON.stringify(value)}`,
    );
  }
  return lines.join("\n");
}

/** The base pipeline with the suggested routes, named after the experiment. */
export function routingDocument(
  base: Record<string, unknown>,
  route: Routing,
  experimentName: string,
): Record<string, unknown> {
  const document = JSON.parse(JSON.stringify(base)) as Record<string, unknown>;
  const stages = (document.stages ??= {}) as Record<string, unknown>;
  const parse = (stages.parse ??= {}) as Record<string, unknown>;
  const routes = (parse.route ??= {}) as Record<string, unknown>;
  for (const [type, value] of routable(route)) routes[type] = value;
  document.name = `${experimentName.slice(0, 56)}-routing`;
  document.description = `The best ${route.path} per gold file type, from experiment ${experimentName}`;
  return document;
}
