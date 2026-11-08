import assert from "node:assert/strict";
import { test } from "node:test";
import {
  applyEvent,
  conversationFor,
  emptyAnswer,
  passageLabel,
  splitCitations,
  streamChat,
  toChatEvent,
  type ChatEvent,
  type ChatPassage,
  type Turn,
} from "../src/lab/chat";

const passage: ChatPassage = {
  number: 1,
  chunk_id: "c1",
  document_id: "d1",
  entry: "operations/espresso-standards.md",
  heading_path: ["3 Recipes", "3.1 Espresso"],
  page_start: null,
  page_end: null,
  score: 0.92,
  text: "Dose 18 g, yield 36 g.",
};

test("citations become parts, and numbers naming no passage stay text", () => {
  assert.deepEqual(splitCitations("Use 18 g [1][2], not [7].", 2), [
    { kind: "text", text: "Use 18 g " },
    { kind: "cite", number: 1 },
    { kind: "cite", number: 2 },
    { kind: "text", text: ", not [7]." },
  ]);
  assert.deepEqual(splitCitations("No sources.", 0), [
    { kind: "text", text: "No sources." },
  ]);
});

test("events build an answer: sources first, then text, then the result", () => {
  const events: ChatEvent[] = [
    {
      kind: "retrieval",
      retrieval: {
        run_id: "r1",
        revision: "abc",
        latency_ms: 12,
        passages: [passage],
      },
    },
    { kind: "delta", text: "Dose " },
    { kind: "delta", text: "18 g [1]." },
    {
      kind: "done",
      done: {
        model: "qwen2.5:3b",
        revision: "357c",
        finish_reason: "stop",
        citations: [1],
        usage: { prompt_tokens: 90, completion_tokens: 6, total_tokens: 96 },
        timings: {
          retrieval_ms: 12,
          ttft_ms: 400,
          generation_ms: 900,
          total_ms: 912,
          load_ms: 1,
          completion_tokens_per_second: 41.3,
        },
        prompt_version: "grounded-chat-v1",
      },
    },
  ];
  const answer = events.reduce(applyEvent, emptyAnswer("qwen2.5:3b"));
  assert.equal(answer.text, "Dose 18 g [1].");
  assert.equal(answer.retrieval?.passages.length, 1);
  assert.equal(answer.streaming, false);
  assert.deepEqual(answer.done?.citations, [1]);
});

test("each column keeps its own thread, and failed answers drop out", () => {
  const turns: Turn[] = [
    {
      question: "What dose?",
      answers: [
        { ...emptyAnswer("a"), text: "18 g [1].", streaming: false },
        {
          ...emptyAnswer("b"),
          error: { code: "x", detail: "y" },
          streaming: false,
        },
      ],
    },
  ];
  assert.deepEqual(conversationFor(turns, 0, "And yield?"), [
    { role: "user", content: "What dose?" },
    { role: "assistant", content: "18 g [1]." },
    { role: "user", content: "And yield?" },
  ]);
  assert.deepEqual(conversationFor(turns, 1, "And yield?"), [
    { role: "user", content: "And yield?" },
  ]);
});

test("unknown or malformed events are ignored", () => {
  assert.equal(toChatEvent({ event: "delta", data: { text: 3 } }), null);
  assert.equal(toChatEvent({ event: "ping", data: {} }), null);
  assert.deepEqual(toChatEvent({ event: "error", data: {} }), {
    kind: "error",
    code: "chat_failed",
    detail: "The answer stopped with an error.",
  });
});

test("a passage's label names its document, headings and pages", () => {
  assert.equal(
    passageLabel(passage),
    "operations/espresso-standards.md — 3 Recipes › 3.1 Espresso",
  );
  assert.equal(
    passageLabel({
      ...passage,
      entry: null,
      heading_path: [],
      page_start: 2,
      page_end: 3,
    }),
    "c1 (pp. 2–3)",
  );
});

const host = {
  apiBase: "/api/v1",
  authHeaders: async () => ({ "X-Test": "1" }),
};

function sseResponse(chunks: string[]): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder();
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

test("streamChat reads events split across chunks", async () => {
  const seen: ChatEvent[] = [];
  let sent: RequestInit | undefined;
  await streamChat(
    host,
    { messages: [{ role: "user", content: "Hi" }], model: "m" },
    (event) => seen.push(event),
    new AbortController().signal,
    {
      fetch: async (_url, init) => {
        sent = init;
        return sseResponse([
          'event: delta\ndata: {"text":"He',
          'llo"}\n\nevent: done\ndata: {"model":"m","citations":[]}\n\n',
        ]);
      },
    },
  );
  assert.deepEqual(
    seen.map((event) => event.kind),
    ["delta", "done"],
  );
  assert.equal((seen[0] as { text: string }).text, "Hello");
  assert.equal((sent?.headers as Record<string, string>)["X-Test"], "1");
});

test("a problem response and a cut-off stream both end as errors", async () => {
  const problems: ChatEvent[] = [];
  await streamChat(
    host,
    { messages: [{ role: "user", content: "Hi" }], model: "m" },
    (event) => problems.push(event),
    new AbortController().signal,
    {
      fetch: async () =>
        new Response(
          JSON.stringify({
            status: 404,
            detail: "Not found.",
            code: "run_not_found",
          }),
          {
            status: 404,
            headers: { "Content-Type": "application/problem+json" },
          },
        ),
    },
  );
  assert.deepEqual(problems, [
    { kind: "error", code: "run_not_found", detail: "Not found." },
  ]);

  const cut: ChatEvent[] = [];
  await streamChat(
    host,
    { messages: [{ role: "user", content: "Hi" }], model: "m" },
    (event) => cut.push(event),
    new AbortController().signal,
    {
      fetch: async () =>
        sseResponse(['event: delta\ndata: {"text":"Hel"}\n\n']),
    },
  );
  assert.equal(cut.at(-1)?.kind, "error");
});
