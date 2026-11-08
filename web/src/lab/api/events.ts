// Live events from GET /events (server-sent events). EventSource can't send
// the identity header, so the stream is read with fetch; parseSse is the pure
// part. Events are hints: screens read state from the API when one arrives.

import type { StratafiHost } from "../host";

export interface LabEvent {
  event: string;
  data: Record<string, unknown>;
}

/**
 * The complete events in `buffer`, and the text after the last one, which
 * belongs to an event still arriving. Comments and malformed data are skipped.
 */
export function parseSse(buffer: string): { events: LabEvent[]; rest: string } {
  const normalized = buffer.replace(/\r\n?/g, "\n");
  const blocks = normalized.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: LabEvent[] = [];
  for (const block of blocks) {
    let event = "message";
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (line === "" || line.startsWith(":")) continue;
      const colon = line.indexOf(":");
      const field = colon < 0 ? line : line.slice(0, colon);
      const value = colon < 0 ? "" : line.slice(colon + 1).replace(/^ /, "");
      if (field === "event") event = value;
      else if (field === "data") data.push(value);
    }
    if (data.length === 0) continue;
    try {
      const parsed: unknown = JSON.parse(data.join("\n"));
      if (
        parsed !== null &&
        typeof parsed === "object" &&
        !Array.isArray(parsed)
      ) {
        events.push({ event, data: parsed as Record<string, unknown> });
      }
    } catch {
      // Not JSON: not an event this client knows.
    }
  }
  return { events, rest };
}

export interface EventFilter {
  run_id?: string;
  job_id?: string;
}

const RETRY_MILLISECONDS = 3000;

function wait(milliseconds: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, milliseconds);
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        resolve();
      },
      { once: true },
    );
  });
}

/**
 * Call `onEvent` for each event until `signal` aborts, reconnecting (with
 * fresh auth headers) when the stream ends or fails. `onOpen` runs on each
 * connection, so a screen can catch up on what it missed.
 */
export async function streamEvents(
  host: Pick<StratafiHost, "apiBase" | "authHeaders">,
  filter: EventFilter,
  onEvent: (event: LabEvent) => void,
  signal: AbortSignal,
  options: { fetch?: typeof fetch; onOpen?: () => void } = {},
): Promise<void> {
  const fetchImpl = options.fetch ?? fetch;
  const query = new URLSearchParams(
    Object.entries(filter).filter(([, value]) => value) as [string, string][],
  );
  const url = `${host.apiBase.replace(/\/+$/, "")}/events?${query}`;
  while (!signal.aborted) {
    try {
      const response = await fetchImpl(url, {
        headers: { Accept: "text/event-stream", ...(await host.authHeaders()) },
        signal,
      });
      if (response.ok && response.body) {
        options.onOpen?.();
        const reader = response.body
          .pipeThrough(new TextDecoderStream())
          .getReader();
        let buffer = "";
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          const parsed = parseSse(buffer + value);
          buffer = parsed.rest;
          for (const event of parsed.events) onEvent(event);
        }
      }
    } catch {
      // Aborted, or the connection dropped: retry below unless aborted.
    }
    await wait(RETRY_MILLISECONDS, signal);
  }
}
