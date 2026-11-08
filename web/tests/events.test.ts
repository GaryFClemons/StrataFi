import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { parseSse, streamEvents, type LabEvent } from "../src/lab/api/events";

describe("server-sent events", () => {
  it("parse complete events and keep the rest for the next read", () => {
    const { events, rest } = parseSse(
      'retry: 3000\n: connected\n\nid: 1\nevent: run\ndata: {"run_id":"r1","status":"running"}\n\nid: 2\nevent: job\ndata: {"jo',
    );
    assert.deepEqual(events, [
      { event: "run", data: { run_id: "r1", status: "running" } },
    ]);
    assert.equal(rest, 'id: 2\nevent: job\ndata: {"jo');
    const next = parseSse(`${rest}b_id":"j1"}\n\n`);
    assert.deepEqual(next.events, [{ event: "job", data: { job_id: "j1" } }]);
    assert.equal(next.rest, "");
  });

  it("handle CRLF, data split over lines, and the default event name", () => {
    const { events } = parseSse('data: {"a":\r\ndata: 1}\r\n\r\n');
    assert.deepEqual(events, [{ event: "message", data: { a: 1 } }]);
  });

  it("skip comments, keepalives and data that isn't a JSON object", () => {
    const { events } = parseSse(
      ": keepalive\n\nevent: lagged\ndata: {}\n\ndata: not json\n\ndata: [1]\n\n",
    );
    assert.deepEqual(events, [{ event: "lagged", data: {} }]);
  });

  it("stream with the host's headers and stop when aborted", async () => {
    const controller = new AbortController();
    const seen: LabEvent[] = [];
    const requests: { url: string; headers: Record<string, string> }[] = [];
    const encoder = new TextEncoder();
    const fakeFetch = (async (url: string, init: RequestInit) => {
      requests.push({ url, headers: init.headers as Record<string, string> });
      const body = new ReadableStream<Uint8Array>({
        start(stream) {
          stream.enqueue(encoder.encode('event: run\ndata: {"run_id":"r1"'));
          stream.enqueue(encoder.encode(',"status":"succeeded"}\n\n'));
          stream.close();
        },
      });
      return new Response(body, { status: 200 });
    }) as unknown as typeof fetch;
    let opened = 0;
    const done = streamEvents(
      {
        apiBase: "/api/v1/",
        authHeaders: async () => ({ "X-Stratafi-Identity": "token" }),
      },
      { run_id: "r1" },
      (event) => {
        seen.push(event);
        controller.abort();
      },
      controller.signal,
      { fetch: fakeFetch, onOpen: () => (opened += 1) },
    );
    await done;
    assert.deepEqual(seen, [
      { event: "run", data: { run_id: "r1", status: "succeeded" } },
    ]);
    assert.equal(opened, 1);
    assert.equal(requests[0]?.url, "/api/v1/events?run_id=r1");
    assert.equal(requests[0]?.headers["X-Stratafi-Identity"], "token");
  });
});
