import { Suspense, lazy, useMemo } from "react";
import { createApiClient } from "./api/client";
import { Empty, Loading, Section } from "./components/ui";
import { LabContext, Link, useLabPath } from "./context";
import type { StratafiHost } from "./host";
import { NAVIGATION, parseRoute, sectionOf, type Route } from "./routes";
import { Corpus, Documents } from "./screens/Documents";
import { EvalCompare, EvalQuery, EvalResults, Evals } from "./screens/Evals";
import { AnswerRunDetail } from "./screens/Answers";
import { EvalSetDetail, EvalSets } from "./screens/EvalSets";
import { ExperimentDetail, Experiments } from "./screens/Experiments";
import { ProfileDetail, Profiles } from "./screens/Profiles";
import { Home } from "./screens/Home";
import { Pipelines } from "./screens/Pipelines";
import { RunDetail, Runs } from "./screens/Runs";
import { Search } from "./screens/Search";
import { Automation, SuiteDetail, TestRunDetail, Tests } from "./screens/Tests";
import {
  TrainedModelDetail,
  Training,
  TrainingRunDetail,
  TrainingSetDetail,
} from "./screens/Training";

// These screens load on first use, which keeps the entry bundle inside the UI budgets.
const Chat = lazy(() => import("./screens/Chat"));
const Compare = lazy(() => import("./screens/Compare"));
const Models = lazy(() => import("./screens/Models"));
const Maps = lazy(() => import("./screens/Maps"));
const MapView = lazy(() =>
  import("./screens/Maps").then((module) => ({ default: module.MapView })),
);

export interface StratafiLabProps {
  host: StratafiHost;
}

function Screen({ route }: { route: Route }) {
  switch (route.name) {
    case "home":
      return <Home />;
    case "documents":
      return <Documents />;
    case "corpus":
      return <Corpus key={route.corpusId} corpusId={route.corpusId} />;
    case "pipelines":
      return <Pipelines step={route.step} />;
    case "runs":
      return <Runs />;
    case "run":
      return <RunDetail key={route.runId} runId={route.runId} />;
    case "search":
      return <Search />;
    case "chat":
      return (
        <Suspense fallback={<Loading label="Loading Chat" />}>
          <Chat />
        </Suspense>
      );
    case "evalsets":
      return <EvalSets />;
    case "evalset":
      return (
        <EvalSetDetail key={route.evalSetId} evalSetId={route.evalSetId} />
      );
    case "evals":
      return <Evals />;
    case "eval":
      return <EvalResults key={route.evalRunId} evalRunId={route.evalRunId} />;
    case "eval-compare":
      return (
        <EvalCompare key={`${route.a}:${route.b}`} a={route.a} b={route.b} />
      );
    case "compare":
      return (
        <Suspense fallback={<Loading label="Loading the Compare workspace" />}>
          <Compare key={route.runs.join("/")} runs={route.runs} />
        </Suspense>
      );
    case "models":
      return (
        <Suspense fallback={<Loading label="Loading Models" />}>
          <Models />
        </Suspense>
      );
    case "maps":
      return (
        <Suspense fallback={<Loading label="Loading Maps" />}>
          <Maps />
        </Suspense>
      );
    case "map":
      return (
        <Suspense fallback={<Loading label="Loading the map" />}>
          <MapView
            key={`${route.pipeline}:${route.retrieval}`}
            pipeline={route.pipeline}
            retrieval={route.retrieval}
          />
        </Suspense>
      );
    case "answers":
      return (
        <AnswerRunDetail
          key={route.answerRunId}
          answerRunId={route.answerRunId}
        />
      );
    case "eval-query":
      return (
        <EvalQuery
          key={`${route.evalRunId}:${route.itemId}`}
          evalRunId={route.evalRunId}
          itemId={route.itemId}
        />
      );
    case "experiments":
      return <Experiments />;
    case "experiment":
      return (
        <ExperimentDetail
          key={route.experimentId}
          experimentId={route.experimentId}
        />
      );
    case "training":
      return <Training />;
    case "training-set":
      return (
        <TrainingSetDetail
          key={route.trainingSetId}
          trainingSetId={route.trainingSetId}
        />
      );
    case "training-run":
      return (
        <TrainingRunDetail
          key={route.trainingRunId}
          trainingRunId={route.trainingRunId}
        />
      );
    case "trained-model":
      return <TrainedModelDetail key={route.modelId} modelId={route.modelId} />;
    case "profiles":
      return <Profiles />;
    case "profile":
      return (
        <ProfileDetail key={route.profileId} profileId={route.profileId} />
      );
    case "tests":
      return <Tests />;
    case "suite":
      return <SuiteDetail key={route.suiteName} suiteName={route.suiteName} />;
    case "test-run":
      return (
        <TestRunDetail key={route.testRunId} testRunId={route.testRunId} />
      );
    case "automation":
      return <Automation />;
    case "not-found":
      return (
        <Section title="Not found">
          <Empty>
            There's no screen here.{" "}
            <Link to={{ name: "home" }} className="text-link">
              Go home
            </Link>
            .
          </Empty>
        </Section>
      );
  }
}

/**
 * The embeddable root. It renders inside whatever frame the host
 * gives it and depends on nothing outside the host interface.
 */
export function StratafiLab({ host }: StratafiLabProps) {
  const api = useMemo(() => createApiClient(host), [host]);
  const lab = useMemo(() => ({ host, api }), [host, api]);
  const route = parseRoute(useLabPath(host.basePath));
  const section = sectionOf(route);

  const { Frame } = host;
  return (
    <LabContext.Provider value={lab}>
      <Frame title="Stratafi">
        <div className="grid gap-4">
          <nav
            aria-label="Stratafi"
            className="flex flex-wrap gap-1 border-b border-separator pb-2"
          >
            {NAVIGATION.map((item) => (
              <Link
                key={item.section}
                to={item.route}
                current={item.section === section}
                className={`rounded-md px-3 py-1 text-sm ${
                  item.section === section
                    ? "bg-default font-semibold"
                    : "hover:bg-default"
                }`}
              >
                {item.label}
              </Link>
            ))}
          </nav>
          <Screen route={route} />
        </div>
      </Frame>
    </LabContext.Provider>
  );
}
