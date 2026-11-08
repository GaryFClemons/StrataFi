import createClient, { type Client, type Middleware } from "openapi-fetch";
import type { StratafiHost } from "../host";
import type { components, paths } from "./generated/schema";

export type ApiClient = Client<paths>;
export type ProblemDetails = components["schemas"]["ProblemDetails"];

/**
 * A typed client for the Stratafi API. Its types are generated from
 * contracts/openapi.json (`npm run api:generate`), and its paths are relative
 * to host.apiBase, so it works wherever the host mounts the API. Each call
 * carries the host's auth headers, fetched fresh.
 */
export function createApiClient(
  host: Pick<StratafiHost, "apiBase" | "authHeaders">,
  options: { fetch?: typeof fetch } = {},
): ApiClient {
  const client = createClient<paths>({
    baseUrl: host.apiBase.replace(/\/+$/, ""),
    fetch: options.fetch,
  });
  const auth: Middleware = {
    async onRequest({ request }) {
      for (const [name, value] of Object.entries(await host.authHeaders())) {
        request.headers.set(name, value);
      }
      return request;
    },
  };
  client.use(auth);
  return client;
}
