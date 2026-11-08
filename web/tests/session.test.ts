import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  DevSession,
  LOCAL_OWNER,
  SIGN_IN_FAILED,
  SignInUnavailable,
  signInLocally,
} from "../src/standalone/session";

function memoryStorage() {
  const items = new Map<string, string>();
  return {
    getItem: (key: string) => items.get(key) ?? null,
    setItem: (key: string, value: string) => void items.set(key, value),
    removeItem: (key: string) => void items.delete(key),
  };
}

function tokenServer(status = 200) {
  const calls: unknown[] = [];
  const fetchImpl = (async (url: string, init: RequestInit) => {
    calls.push({ url, body: JSON.parse(String(init.body)) });
    if (status !== 200) return new Response("{}", { status });
    return Response.json({
      token: `token-${calls.length}`,
      expires_in: 300,
      subject: "u-tenant0-editor",
    });
  }) as unknown as typeof fetch;
  return { calls, fetchImpl };
}

describe("the development session", () => {
  it("signs in, keeps the identity for the tab, and sends a token", async () => {
    const storage = memoryStorage();
    const server = tokenServer();
    const session = new DevSession(
      "/api/v1/",
      storage,
      server.fetchImpl,
      () => 0,
    );
    assert.deepEqual(await session.headers(), {});
    await session.signIn({ tenant: "tenant0", role: "editor" });
    assert.deepEqual(session.identity(), { tenant: "tenant0", role: "editor" });
    assert.deepEqual(await session.headers(), {
      "X-Stratafi-Identity": "token-1",
    });
    assert.deepEqual(server.calls, [
      { url: "/api/v1/dev/token", body: { tenant: "tenant0", role: "editor" } },
    ]);
  });

  it("refreshes the token a minute before it expires", async () => {
    let now = 0;
    const server = tokenServer();
    const session = new DevSession(
      "/api/v1",
      memoryStorage(),
      server.fetchImpl,
      () => now,
    );
    await session.signIn({ tenant: "tenant0", role: "viewer" });
    now = 239_000;
    assert.deepEqual(await session.headers(), {
      "X-Stratafi-Identity": "token-1",
    });
    now = 240_000;
    assert.deepEqual(await session.headers(), {
      "X-Stratafi-Identity": "token-2",
    });
  });

  it("fetches one token for calls made together", async () => {
    const storage = memoryStorage();
    storage.setItem(
      "stratafi.dev-identity",
      JSON.stringify({ tenant: "t", role: "owner" }),
    );
    const server = tokenServer();
    const session = new DevSession(
      "/api/v1",
      storage,
      server.fetchImpl,
      () => 0,
    );
    const headers = await Promise.all([session.headers(), session.headers()]);
    assert.equal(server.calls.length, 1);
    assert.deepEqual(headers[0], headers[1]);
  });

  it("says when the API has sign-in off", async () => {
    const session = new DevSession(
      "/api/v1",
      memoryStorage(),
      tokenServer(404).fetchImpl,
    );
    await assert.rejects(
      session.signIn({ tenant: "tenant0", role: "editor" }),
      SignInUnavailable,
    );
    assert.equal(session.identity(), null);
  });

  it("signs out, and survives storage it can't use", async () => {
    const storage = memoryStorage();
    const session = new DevSession("/api/v1", storage, tokenServer().fetchImpl);
    await session.signIn({ tenant: "tenant0", role: "editor" });
    session.signOut();
    assert.equal(session.identity(), null);
    assert.deepEqual(await session.headers(), {});

    const broken = {
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("blocked");
      },
      removeItem: () => undefined,
    };
    const fallback = new DevSession("/api/v1", broken, tokenServer().fetchImpl);
    await fallback.signIn({ tenant: "tenant0", role: "editor" });
    assert.equal(fallback.identity(), null);
    assert.deepEqual(await fallback.headers(), {
      "X-Stratafi-Identity": "token-1",
    });
  });
});

/** The API: GET /auth/config says whether sign-in is on; POST /dev/token mints. */
function api(config: { status: number; on?: boolean } | "down") {
  const server = tokenServer();
  const urls: string[] = [];
  const fetchImpl = (async (url: string, init?: RequestInit) => {
    urls.push(url);
    if (url.endsWith("/auth/config")) {
      if (config === "down") throw new TypeError("failed to fetch");
      return Response.json({ dev_sign_in: config.on === true }, config);
    }
    return server.fetchImpl(url, init);
  }) as unknown as typeof fetch;
  return { urls, calls: server.calls, fetchImpl };
}

describe("signing in locally", () => {
  it("signs in as the local owner when the API's sign-in is on", async () => {
    const lab = api({ status: 200, on: true });
    const session = new DevSession("/api/v1", memoryStorage(), lab.fetchImpl);
    assert.equal(await signInLocally(session, "/api/v1/", lab.fetchImpl), null);
    assert.deepEqual(session.identity(), LOCAL_OWNER);
    assert.deepEqual(LOCAL_OWNER, { tenant: "local", role: "owner" });
    assert.deepEqual(lab.calls, [
      { url: "/api/v1/dev/token", body: { tenant: "local", role: "owner" } },
    ]);
    assert.equal(session.label(), "owner in local (development)");
  });

  it("keeps whoever the tab signed in as", async () => {
    const storage = memoryStorage();
    storage.setItem(
      "stratafi.dev-identity",
      JSON.stringify({ tenant: "tenant0", role: "viewer" }),
    );
    const lab = api({ status: 200, on: true });
    const session = new DevSession("/api/v1", storage, lab.fetchImpl);
    assert.equal(await signInLocally(session, "/api/v1", lab.fetchImpl), null);
    assert.deepEqual(lab.urls, []);
    assert.deepEqual(session.identity(), { tenant: "tenant0", role: "viewer" });
  });

  it("says why it couldn't", async () => {
    for (const [config, expected] of [
      [{ status: 200, on: false }, new SignInUnavailable().message],
      [{ status: 503 }, SIGN_IN_FAILED],
      ["down", SIGN_IN_FAILED],
    ] as const) {
      const lab = api(config);
      const session = new DevSession("/api/v1", memoryStorage(), lab.fetchImpl);
      assert.equal(
        await signInLocally(session, "/api/v1", lab.fetchImpl),
        expected,
      );
      assert.equal(session.identity(), null);
      assert.deepEqual(lab.calls, []);
    }
  });
});
