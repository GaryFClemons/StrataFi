import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import type { ProblemDetails } from "./client";

/** A failed API call: its problem's code and a message that never echoes input. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string | null;

  constructor(status: number, problem: Partial<ProblemDetails> | null) {
    const message =
      problem?.detail ?? problem?.title ?? `The request failed (${status}).`;
    super(message);
    this.status = status;
    this.code = (problem?.code as string | null | undefined) ?? null;
  }
}

/** openapi-fetch's result as its data, or an ApiError. */
export function unwrap<T>(result: {
  data?: T;
  error?: unknown;
  response: Response;
}): T {
  if (result.error !== undefined || result.data === undefined) {
    const problem =
      result.error !== null && typeof result.error === "object"
        ? (result.error as Partial<ProblemDetails>)
        : null;
    throw new ApiError(result.response.status, problem);
  }
  return result.data;
}

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  return "Something went wrong.";
}

export interface Loaded<T> {
  data: T | undefined;
  error: unknown;
  loading: boolean;
  reload: () => void;
}

interface LoadState<T> {
  key: string;
  request: string;
  data?: T;
  error?: unknown;
}

/**
 * Load data for `key`, again whenever it changes; `reload` loads it again in
 * place, keeping what's shown until the new data arrives. Responses for an
 * older request are dropped.
 */
export function useLoad<T>(load: () => Promise<T>, key: string): Loaded<T> {
  const [state, setState] = useState<LoadState<T> | null>(null);
  const [tick, setTick] = useState(0);
  const loadRef = useRef(load);
  useLayoutEffect(() => {
    loadRef.current = load;
  });
  const request = `${key}\n${tick}`;

  useEffect(() => {
    let current = true;
    loadRef.current().then(
      (data) => {
        if (current) setState({ key, request, data });
      },
      (error: unknown) => {
        if (current) {
          setState((previous) => ({
            key,
            request,
            data: previous?.key === key ? previous.data : undefined,
            error,
          }));
        }
      },
    );
    return () => {
      current = false;
    };
  }, [key, request]);

  const reload = useCallback(() => setTick((value) => value + 1), []);
  const same = state?.key === key;
  return {
    data: same ? state.data : undefined,
    error: same && state.request === request ? state.error : undefined,
    loading: state?.request !== request,
    reload,
  };
}
