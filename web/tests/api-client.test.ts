import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { createApiClient } from "../src/lab/api";

function fakeFetch(calls: Request[], body: unknown, status = 200) {
  return async (input: Request | URL | string, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(input, init);
    calls.push(request);
    return new Response(JSON.stringify(body), {
      status,
      headers: {
        "content-type":
          status >= 400 ? "application/problem+json" : "application/json",
      },
    });
  };
}

describe("the generated API client", () => {
  it("joins paths to the host's API base and sends its auth headers", async () => {
    const calls: Request[] = [];
    const client = createApiClient(
      {
        apiBase: "http://lab.test/api/v1/",
        authHeaders: async () => ({ "X-Stratafi-Identity": "token-1" }),
      },
      {
        fetch: fakeFetch(calls, {
          status: "ok",
          version: "0.1.0",
          checks: {},
        }),
      },
    );
    const { data, error } = await client.GET("/health");
    assert.equal(error, undefined);
    assert.equal(data?.status, "ok");
    assert.equal(calls[0].url, "http://lab.test/api/v1/health");
    assert.equal(calls[0].headers.get("X-Stratafi-Identity"), "token-1");
  });

  it("returns problem details as the error", async () => {
    const calls: Request[] = [];
    const client = createApiClient(
      { apiBase: "http://lab.test/api/v1", authHeaders: async () => ({}) },
      {
        fetch: fakeFetch(
          calls,
          {
            type: "urn:stratafi:problem:unauthorized",
            title: "Unauthorized",
            status: 401,
            detail: "A valid identity token is required.",
            code: "missing",
          },
          401,
        ),
      },
    );
    const { data, error } = await client.GET("/me");
    assert.equal(data, undefined);
    assert.equal(error?.code, "missing");
  });

  it("sends query parameters for paged lists", async () => {
    const calls: Request[] = [];
    const client = createApiClient(
      { apiBase: "http://lab.test/api/v1", authHeaders: async () => ({}) },
      { fetch: fakeFetch(calls, { items: [], next_cursor: null }) },
    );
    await client.GET("/audit-events", {
      params: { query: { limit: 10, cursor: "abc" } },
    });
    const url = new URL(calls[0].url);
    assert.equal(url.pathname, "/api/v1/audit-events");
    assert.equal(url.searchParams.get("limit"), "10");
    assert.equal(url.searchParams.get("cursor"), "abc");
  });
});
