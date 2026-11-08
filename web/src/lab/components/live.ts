import { useEffect } from "react";
import type { components } from "../api";
import { streamEvents } from "../api/events";
import { unwrap, useLoad, type Loaded } from "../api/load";
import { useLab } from "../context";
import { isExperimentActive } from "../experiments";
import { isActive } from "../format";

export type RunOut = components["schemas"]["RunOut"];

const THROTTLE_MILLISECONDS = 250;

/**
 * A run, kept current while it's active: its `run` events (GET /events?run_id=)
 * trigger a reload, at most a few times a second, and listening stops once
 * the run has finished.
 */
export function useLiveRun(runId: string | null): Loaded<RunOut> {
  const { api, host } = useLab();
  const loaded = useLoad<RunOut>(async () => {
    if (runId === null) throw new Error("no run");
    return unwrap(
      await api.GET("/runs/{run_id}", { params: { path: { run_id: runId } } }),
    );
  }, runId ?? "");
  const active = loaded.data ? isActive(loaded.data.status) : runId !== null;
  const { reload } = loaded;

  useEffect(() => {
    if (runId === null || !active) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | null = null;
    const schedule = () => {
      timer ??= setTimeout(() => {
        timer = null;
        reload();
      }, THROTTLE_MILLISECONDS);
    };
    void streamEvents(host, { run_id: runId }, schedule, controller.signal, {
      onOpen: schedule,
    });
    return () => {
      controller.abort();
      if (timer !== null) clearTimeout(timer);
    };
  }, [host, runId, active, reload]);

  return loaded;
}

export type EvalRunOut = components["schemas"]["EvalRunOut"];

/** An eval run, kept current while it's active, as useLiveRun keeps a run. */
export function useLiveEvalRun(evalRunId: string): Loaded<EvalRunOut> {
  const { api, host } = useLab();
  const loaded = useLoad<EvalRunOut>(
    async () =>
      unwrap(
        await api.GET("/eval-runs/{eval_run_id}", {
          params: { path: { eval_run_id: evalRunId } },
        }),
      ),
    evalRunId,
  );
  const active = loaded.data ? isActive(loaded.data.status) : true;
  const { reload } = loaded;

  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | null = null;
    const schedule = () => {
      timer ??= setTimeout(() => {
        timer = null;
        reload();
      }, THROTTLE_MILLISECONDS);
    };
    void streamEvents(
      host,
      { run_id: evalRunId },
      schedule,
      controller.signal,
      {
        onOpen: schedule,
      },
    );
    return () => {
      controller.abort();
      if (timer !== null) clearTimeout(timer);
    };
  }, [host, evalRunId, active, reload]);

  return loaded;
}

export type ExperimentOut = components["schemas"]["ExperimentOut"];

/**
 * An experiment, kept current while it runs: its `experiment` events (with its
 * ID as `run_id`) trigger a reload, as useLiveRun does for a run.
 */
export function useLiveExperiment(experimentId: string): Loaded<ExperimentOut> {
  const { api, host } = useLab();
  const loaded = useLoad<ExperimentOut>(
    async () =>
      unwrap(
        await api.GET("/experiments/{experiment_id}", {
          params: { path: { experiment_id: experimentId } },
        }),
      ),
    experimentId,
  );
  const active = loaded.data ? isExperimentActive(loaded.data.status) : true;
  const { reload } = loaded;

  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | null = null;
    const schedule = () => {
      timer ??= setTimeout(() => {
        timer = null;
        reload();
      }, THROTTLE_MILLISECONDS);
    };
    void streamEvents(
      host,
      { run_id: experimentId },
      schedule,
      controller.signal,
      { onOpen: schedule },
    );
    return () => {
      controller.abort();
      if (timer !== null) clearTimeout(timer);
    };
  }, [host, experimentId, active, reload]);

  return loaded;
}

export type TestRunOut = components["schemas"]["TestRunOut"];

const TEST_RUN_FINISHED = ["passed", "failed", "error", "cancelled"];

/**
 * A test run, kept current while it's queued, running or evaluating: its
 * `test_run` events (with its ID as `run_id`) trigger a reload.
 */
export function useLiveTestRun(testRunId: string): Loaded<TestRunOut> {
  const { api, host } = useLab();
  const loaded = useLoad<TestRunOut>(
    async () =>
      unwrap(
        await api.GET("/test-runs/{test_run_id}", {
          params: { path: { test_run_id: testRunId } },
        }),
      ),
    testRunId,
  );
  const active = loaded.data
    ? !TEST_RUN_FINISHED.includes(loaded.data.status)
    : true;
  const { reload } = loaded;

  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | null = null;
    const schedule = () => {
      timer ??= setTimeout(() => {
        timer = null;
        reload();
      }, THROTTLE_MILLISECONDS);
    };
    void streamEvents(
      host,
      { run_id: testRunId },
      schedule,
      controller.signal,
      { onOpen: schedule },
    );
    return () => {
      controller.abort();
      if (timer !== null) clearTimeout(timer);
    };
  }, [host, testRunId, active, reload]);

  return loaded;
}
