// The Chat screen's client: POST /chat streams server-sent events
// (retrieval, delta, done, error), read with fetch since EventSource can't
// POST or send the identity header. Everything but streamChat and
// loadChatModels is pure, so the screen's logic is tested without a server.

import { parseSse, type LabEvent } from "./api/events";
import { ApiError } from "./api/load";
import type { StratafiHost } from "./host";

export type RetrievalMode = "hybrid" | "dense" | "keyword";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  model: string;
  run_id?: string | null;
  mode?: RetrievalMode;
  retrieval_config_id?: string | null;
  top_k?: number;
  temperature?: number;
  max_tokens?: number;
  seed?: number | null;
  system_prompt?: string | null;
}

export interface ChatPassage {
  number: number;
  chunk_id: string;
  document_id: string;
  entry: string | null;
  heading_path: string[];
  page_start: number | null;
  page_end: number | null;
  score: number;
  text: string;
}

export interface ChatRetrieval {
  run_id: string;
  revision: string | null;
  latency_ms: number;
  passages: ChatPassage[];
}

export interface ChatTimings {
  retrieval_ms: number | null;
  ttft_ms: number | null;
  generation_ms: number;
  total_ms: number;
  load_ms: number | null;
  completion_tokens_per_second: number | null;
}

export interface ChatUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ChatDone {
  model: string;
  revision: string | null;
  finish_reason: string;
  citations: number[];
  usage: ChatUsage | null;
  timings: ChatTimings;
  prompt_version: string;
}

export interface ChatModel {
  id: string;
  revision: string | null;
  served_model: string | null;
  parameter_size: string | null;
  quantization: string | null;
  context_length: number | null;
}

export type ChatEvent =
  | { kind: "retrieval"; retrieval: ChatRetrieval }
  | { kind: "delta"; text: string }
  | { kind: "done"; done: ChatDone }
  | { kind: "error"; code: string; detail: string };

/** One model's answer to one question, as it streams in. */
export interface Answer {
  model: string;
  text: string;
  retrieval: ChatRetrieval | null;
  done: ChatDone | null;
  error: { code: string; detail: string } | null;
  streaming: boolean;
}

/** A question and its answers: one per model being compared. */
export interface Turn {
  question: string;
  answers: Answer[];
}

export function emptyAnswer(model: string): Answer {
  return {
    model,
    text: "",
    retrieval: null,
    done: null,
    error: null,
    streaming: true,
  };
}

/** The answer after one more event. */
export function applyEvent(answer: Answer, event: ChatEvent): Answer {
  switch (event.kind) {
    case "retrieval":
      return { ...answer, retrieval: event.retrieval };
    case "delta":
      return { ...answer, text: answer.text + event.text };
    case "done":
      return { ...answer, done: event.done, streaming: false };
    case "error":
      return {
        ...answer,
        error: { code: event.code, detail: event.detail },
        streaming: false,
      };
  }
}

/** A server-sent event as a chat event, or null if it isn't one. */
export function toChatEvent(raw: LabEvent): ChatEvent | null {
  const data = raw.data;
  switch (raw.event) {
    case "retrieval":
      return Array.isArray(data.passages)
        ? { kind: "retrieval", retrieval: data as unknown as ChatRetrieval }
        : null;
    case "delta":
      return typeof data.text === "string"
        ? { kind: "delta", text: data.text }
        : null;
    case "done":
      return typeof data.model === "string"
        ? { kind: "done", done: data as unknown as ChatDone }
        : null;
    case "error":
      return {
        kind: "error",
        code: typeof data.code === "string" ? data.code : "chat_failed",
        detail:
          typeof data.detail === "string"
            ? data.detail
            : "The answer stopped with an error.",
      };
    default:
      return null;
  }
}

/**
 * The messages to send for one model: each earlier question with that model's
 * own answer (so each side of a comparison keeps its own thread), then the
 * new question. Failed or empty answers are left out.
 */
export function conversationFor(
  turns: Turn[],
  column: number,
  question: string,
): ChatMessage[] {
  const messages: ChatMessage[] = [];
  for (const turn of turns) {
    const answer = turn.answers[column];
    if (!answer || answer.error || !answer.text.trim()) continue;
    messages.push({ role: "user", content: turn.question });
    messages.push({ role: "assistant", content: answer.text });
  }
  messages.push({ role: "user", content: question });
  return messages;
}

export type AnswerPart =
  { kind: "text"; text: string } | { kind: "cite"; number: number };

/**
 * An answer's text with its citations, like [2], picked out so each can link
 * to its passage. Numbers that name no passage stay text.
 */
export function splitCitations(text: string, passages: number): AnswerPart[] {
  const parts: AnswerPart[] = [];
  let last = 0;
  for (const match of text.matchAll(/\[(\d{1,3})\]/g)) {
    const number = Number(match[1]);
    if (number < 1 || number > passages) continue;
    const at = match.index ?? 0;
    if (at > last) parts.push({ kind: "text", text: text.slice(last, at) });
    parts.push({ kind: "cite", number });
    last = at + match[0].length;
  }
  if (last < text.length) parts.push({ kind: "text", text: text.slice(last) });
  return parts;
}

/** A passage's place: its document, heading path and pages. */
export function passageLabel(passage: ChatPassage): string {
  const place = [passage.entry, passage.heading_path.join(" › ")]
    .filter(Boolean)
    .join(" — ");
  const pages =
    passage.page_start === null
      ? ""
      : passage.page_end !== null && passage.page_end !== passage.page_start
        ? ` (pp. ${passage.page_start}–${passage.page_end})`
        : ` (p. ${passage.page_start})`;
  return (place || passage.chunk_id) + pages;
}

async function problemOf(response: Response): Promise<ApiError> {
  const problem = await response
    .json()
    .then((body: unknown) =>
      body !== null && typeof body === "object"
        ? (body as Record<string, unknown>)
        : null,
    )
    .catch(() => null);
  return new ApiError(response.status, problem);
}

function url(host: Pick<StratafiHost, "apiBase">, path: string): string {
  return `${host.apiBase.replace(/\/+$/, "")}${path}`;
}

/** GET /chat/models: the gateway's chat models and its default. */
export async function loadChatModels(
  host: Pick<StratafiHost, "apiBase" | "authHeaders">,
  options: { fetch?: typeof fetch } = {},
): Promise<{ models: ChatModel[]; default: string | null }> {
  const fetchImpl = options.fetch ?? fetch;
  const response = await fetchImpl(url(host, "/chat/models"), {
    headers: { Accept: "application/json", ...(await host.authHeaders()) },
  });
  if (!response.ok) throw await problemOf(response);
  return (await response.json()) as {
    models: ChatModel[];
    default: string | null;
  };
}

/**
 * POST /chat and call `onEvent` for each event until the answer ends or
 * `signal` aborts. A problem before the stream starts (a missing run, the
 * gateway down) arrives as an error event too.
 */
export async function streamChat(
  host: Pick<StratafiHost, "apiBase" | "authHeaders">,
  request: ChatRequest,
  onEvent: (event: ChatEvent) => void,
  signal: AbortSignal,
  options: { fetch?: typeof fetch } = {},
): Promise<void> {
  const fetchImpl = options.fetch ?? fetch;
  let response: Response;
  try {
    response = await fetchImpl(url(host, "/chat"), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
        ...(await host.authHeaders()),
      },
      body: JSON.stringify(request),
      signal,
    });
  } catch {
    if (!signal.aborted) {
      onEvent({
        kind: "error",
        code: "network",
        detail: "The API can't be reached.",
      });
    }
    return;
  }
  if (!response.ok || !response.body) {
    const error = await problemOf(response);
    onEvent({
      kind: "error",
      code: error.code ?? "chat_failed",
      detail: error.message,
    });
    return;
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  let ended = false;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      const parsed = parseSse(buffer + value);
      buffer = parsed.rest;
      for (const raw of parsed.events) {
        const event = toChatEvent(raw);
        if (event === null) continue;
        if (event.kind === "done" || event.kind === "error") ended = true;
        onEvent(event);
      }
    }
  } catch {
    // Aborted, or the connection dropped: said below unless aborted.
  }
  if (!ended && !signal.aborted) {
    onEvent({
      kind: "error",
      code: "stream_ended",
      detail: "The answer stopped before it finished.",
    });
  }
}
