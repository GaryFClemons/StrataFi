// The lab's routes: one hand-written module over lab paths.
// Lab paths are relative to host.basePath (see paths.ts), so the same routes
// work at /stratafi/ standalone and wherever a host mounts the lab.

/** The five steps of a pipeline, then search. */
export const STEP_IDS = [
  "extract",
  "scan",
  "parse",
  "chunk",
  "embed",
  "search",
] as const;
export type StepId = (typeof STEP_IDS)[number];

export type Route =
  | { name: "home" }
  | { name: "documents" }
  | { name: "corpus"; corpusId: string }
  | { name: "pipelines"; step: StepId }
  | { name: "runs" }
  | { name: "run"; runId: string }
  | { name: "search" }
  | { name: "chat" }
  | { name: "evalsets" }
  | { name: "evalset"; evalSetId: string }
  | { name: "evals" }
  | { name: "eval"; evalRunId: string }
  | { name: "eval-query"; evalRunId: string; itemId: string }
  | { name: "eval-compare"; a: string | null; b: string | null }
  | { name: "compare"; runs: string[] }
  | { name: "models" }
  | { name: "maps" }
  | { name: "map"; pipeline: string | null; retrieval: string | null }
  | { name: "answers"; answerRunId: string }
  | { name: "experiments" }
  | { name: "experiment"; experimentId: string }
  | { name: "training" }
  | { name: "training-set"; trainingSetId: string }
  | { name: "training-run"; trainingRunId: string }
  | { name: "trained-model"; modelId: string }
  | { name: "profiles" }
  | { name: "profile"; profileId: string }
  | { name: "tests" }
  | { name: "suite"; suiteName: string }
  | { name: "test-run"; testRunId: string }
  | { name: "automation" }
  | { name: "not-found"; path: string };

export function isStepId(value: string): value is StepId {
  return (STEP_IDS as readonly string[]).includes(value);
}

function segments(labPath: string): string[] {
  const path = labPath.split(/[?#]/, 1)[0] ?? "";
  return path
    .split("/")
    .filter(Boolean)
    .map((segment) => {
      try {
        return decodeURIComponent(segment);
      } catch {
        return segment;
      }
    });
}

/** In a map's path, "-" stands for no pipeline spec or no retrieval config. */
const NONE = "-";

/** The route for a lab path: "/runs/7" → { name: "run", runId: "7" }. */
export function parseRoute(labPath: string): Route {
  const parts = segments(labPath);
  const [first, second, ...rest] = parts;
  if (first === "compare" && second !== undefined) {
    const runs = [second, ...rest];
    if (runs.length <= 6) return { name: "compare", runs };
  }
  if (first === "maps" && second !== undefined && rest.length === 1) {
    const retrieval = rest[0] as string;
    if (second !== NONE || retrieval !== NONE) {
      return {
        name: "map",
        pipeline: second === NONE ? null : second,
        retrieval: retrieval === NONE ? null : retrieval,
      };
    }
  }
  if (rest.length === 0) {
    if (first === undefined) return { name: "home" };
    if (first === "documents") {
      return second === undefined
        ? { name: "documents" }
        : { name: "corpus", corpusId: second };
    }
    if (first === "pipelines") {
      if (second === undefined) return { name: "pipelines", step: "extract" };
      if (isStepId(second)) return { name: "pipelines", step: second };
    }
    if (first === "runs") {
      return second === undefined
        ? { name: "runs" }
        : { name: "run", runId: second };
    }
    if (first === "search" && second === undefined) return { name: "search" };
    if (first === "chat" && second === undefined) return { name: "chat" };
    if (first === "models" && second === undefined) return { name: "models" };
    if (first === "maps" && second === undefined) return { name: "maps" };
    if (first === "eval-sets") {
      return second === undefined
        ? { name: "evalsets" }
        : { name: "evalset", evalSetId: second };
    }
    if (first === "evals") {
      if (second === "compare")
        return { name: "eval-compare", a: null, b: null };
      return second === undefined
        ? { name: "evals" }
        : { name: "eval", evalRunId: second };
    }
    if (first === "experiments") {
      return second === undefined
        ? { name: "experiments" }
        : { name: "experiment", experimentId: second };
    }
    if (first === "profiles") {
      return second === undefined
        ? { name: "profiles" }
        : { name: "profile", profileId: second };
    }
    if (first === "training" && second === undefined) {
      return { name: "training" };
    }
  }
  if (first === "training" && rest.length === 1) {
    const id = rest[0] as string;
    if (second === "sets") return { name: "training-set", trainingSetId: id };
    if (second === "runs") return { name: "training-run", trainingRunId: id };
    if (second === "models") return { name: "trained-model", modelId: id };
  }
  if (first === "tests") {
    if (second === undefined && rest.length === 0) return { name: "tests" };
    if (second === "automation" && rest.length === 0)
      return { name: "automation" };
    if (second === "suites" && rest.length === 1)
      return { name: "suite", suiteName: rest[0] as string };
    if (second === "runs" && rest.length === 1)
      return { name: "test-run", testRunId: rest[0] as string };
  }
  if (first === "evals" && second === "answers" && rest.length === 1) {
    return { name: "answers", answerRunId: rest[0] as string };
  }
  if (first === "evals" && second !== undefined && rest.length === 2) {
    const [third, fourth] = rest;
    if (second === "compare" && third !== undefined && fourth !== undefined) {
      return { name: "eval-compare", a: third, b: fourth };
    }
    if (third === "queries" && fourth !== undefined) {
      return { name: "eval-query", evalRunId: second, itemId: fourth };
    }
  }
  return { name: "not-found", path: `/${parts.join("/")}` };
}

/** The lab path for a route, the inverse of parseRoute. */
export function routePath(route: Route): string {
  switch (route.name) {
    case "home":
      return "/";
    case "documents":
      return "/documents";
    case "corpus":
      return `/documents/${encodeURIComponent(route.corpusId)}`;
    case "pipelines":
      return `/pipelines/${route.step}`;
    case "runs":
      return "/runs";
    case "run":
      return `/runs/${encodeURIComponent(route.runId)}`;
    case "search":
      return "/search";
    case "chat":
      return "/chat";
    case "evalsets":
      return "/eval-sets";
    case "evalset":
      return `/eval-sets/${encodeURIComponent(route.evalSetId)}`;
    case "evals":
      return "/evals";
    case "eval":
      return `/evals/${encodeURIComponent(route.evalRunId)}`;
    case "eval-query":
      return `/evals/${encodeURIComponent(route.evalRunId)}/queries/${encodeURIComponent(route.itemId)}`;
    case "eval-compare":
      return route.a !== null && route.b !== null
        ? `/evals/compare/${encodeURIComponent(route.a)}/${encodeURIComponent(route.b)}`
        : "/evals/compare";
    case "answers":
      return `/evals/answers/${encodeURIComponent(route.answerRunId)}`;
    case "compare":
      return `/compare/${route.runs.map(encodeURIComponent).join("/")}`;
    case "models":
      return "/models";
    case "maps":
      return "/maps";
    case "map":
      return `/maps/${encodeURIComponent(route.pipeline ?? NONE)}/${encodeURIComponent(route.retrieval ?? NONE)}`;
    case "experiments":
      return "/experiments";
    case "experiment":
      return `/experiments/${encodeURIComponent(route.experimentId)}`;
    case "training":
      return "/training";
    case "training-set":
      return `/training/sets/${encodeURIComponent(route.trainingSetId)}`;
    case "training-run":
      return `/training/runs/${encodeURIComponent(route.trainingRunId)}`;
    case "trained-model":
      return `/training/models/${encodeURIComponent(route.modelId)}`;
    case "profiles":
      return "/profiles";
    case "profile":
      return `/profiles/${encodeURIComponent(route.profileId)}`;
    case "tests":
      return "/tests";
    case "suite":
      return `/tests/suites/${encodeURIComponent(route.suiteName)}`;
    case "test-run":
      return `/tests/runs/${encodeURIComponent(route.testRunId)}`;
    case "automation":
      return "/tests/automation";
    case "not-found":
      return route.path;
  }
}

export type Section =
  | "home"
  | "documents"
  | "pipelines"
  | "maps"
  | "models"
  | "runs"
  | "search"
  | "chat"
  | "evalsets"
  | "evals"
  | "experiments"
  | "training"
  | "profiles"
  | "tests";

/** The navigation, in order. */
export const NAVIGATION: { section: Section; label: string; route: Route }[] = [
  { section: "home", label: "Home", route: { name: "home" } },
  { section: "chat", label: "Chat", route: { name: "chat" } },
  { section: "documents", label: "Documents", route: { name: "documents" } },
  {
    section: "pipelines",
    label: "Pipelines",
    route: { name: "pipelines", step: "extract" },
  },
  { section: "maps", label: "Maps", route: { name: "maps" } },
  { section: "runs", label: "Runs", route: { name: "runs" } },
  { section: "search", label: "Search", route: { name: "search" } },
  { section: "evalsets", label: "Eval sets", route: { name: "evalsets" } },
  { section: "evals", label: "Evals", route: { name: "evals" } },
  { section: "models", label: "Models", route: { name: "models" } },
  {
    section: "experiments",
    label: "Experiments",
    route: { name: "experiments" },
  },
  { section: "training", label: "Training", route: { name: "training" } },
  { section: "profiles", label: "Profiles", route: { name: "profiles" } },
  { section: "tests", label: "Tests", route: { name: "tests" } },
];

/** The navigation section a route belongs to, or null. */
export function sectionOf(route: Route): Section | null {
  switch (route.name) {
    case "home":
      return "home";
    case "documents":
    case "corpus":
      return "documents";
    case "pipelines":
      return "pipelines";
    case "runs":
    case "run":
      return "runs";
    case "search":
      return "search";
    case "chat":
      return "chat";
    case "evalsets":
    case "evalset":
      return "evalsets";
    case "evals":
    case "eval":
    case "eval-query":
    case "eval-compare":
    case "compare":
    case "answers":
      return "evals";
    case "models":
      return "models";
    case "maps":
    case "map":
      return "maps";
    case "experiments":
    case "experiment":
      return "experiments";
    case "training":
    case "training-set":
    case "training-run":
    case "trained-model":
      return "training";
    case "profiles":
    case "profile":
      return "profiles";
    case "tests":
    case "suite":
    case "test-run":
    case "automation":
      return "tests";
    case "not-found":
      return null;
  }
}
