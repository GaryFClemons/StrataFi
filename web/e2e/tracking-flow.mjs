// The tracking browser flow: search eval runs with the search
// language, annotate one, select three and compare them (charts, queries side
// by side, maps with their differences), open a query's trace, every RAG
// pipeline's map, and the model comparison. No query text may reach the
// browser console.
//
//   node e2e/tracking-flow.mjs <app URL> <eval run> <eval run> <eval run>
//
// services/worker/tests/test_tracking_integration.py evaluates three retrieval
// configs over the Halcyon demo corpus first, serves the built app beside the API,
// and runs this.

import assert from "node:assert/strict";
import { open, signIn } from "./lib.mjs";

const [appUrl, ...runs] = process.argv.slice(2);
if (!appUrl || runs.length !== 3) {
  console.error("usage: node e2e/tracking-flow.mjs <app URL> <3 eval run IDs>");
  process.exit(2);
}

const { browser, page, console_, step } = await open();
const nav = (name) =>
  page
    .getByRole("navigation", { name: "Stratafi" })
    .getByRole("link", { name, exact: true });
const rendered = (testId) =>
  page.locator(`[data-testid="${testId}"][data-rendered="true"]`).first();
const at = (path) =>
  new URL(path, appUrl.endsWith("/") ? appUrl : `${appUrl}/`).href;

try {
  await step("sign in as an editor");
  await signIn(page, appUrl);

  await step("search eval runs with the search language");
  await nav("Evals").click();
  const table = page.getByRole("table", { name: "Eval runs", exact: true });
  await table.getByRole("row").nth(3).waitFor();
  const filter = page.getByLabel("Filter", { exact: true });
  await filter.fill("params.rerank.component = 'rerank/none'");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await page.getByTestId("search-total").getByText("2 of 3").waitFor();
  await filter.fill("metrics.recall@5 >");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await page
    .getByRole("alert")
    .getByText(/doesn't parse/)
    .waitFor();
  await filter.fill("");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await page.getByTestId("search-total").getByText("3 of 3").waitFor();
  await rendered("tracking-chart").waitFor();

  await step("annotate an eval run");
  await page.goto(at(`evals/${runs[0]}`));
  await page.getByRole("button", { name: "Edit", exact: true }).click();
  const form = page.getByRole("form", { name: "Annotate" });
  await form.getByLabel("Title").fill("Hybrid with the gateway reranker");
  await form.getByLabel(/^Tags/).fill("team=rag, stage=baseline");
  await form
    .getByLabel(/^Note/)
    .fill("# Why\n\nThe **baseline** for tracking.");
  await form.getByRole("button", { name: "Save" }).click();
  await page.getByTestId("tag").filter({ hasText: "team=rag" }).waitFor();
  await page.getByTestId("note").getByText("baseline").waitFor();

  await step("open a query's trace from the slowest queries");
  const slowest = page.getByRole("table", { name: "Slowest queries" });
  await slowest.getByRole("link").first().click();
  await rendered("trace-waterfall").waitFor();
  const spans = page.getByRole("table", { name: "Spans" });
  await spans.getByRole("cell", { name: "Rerank", exact: true }).waitFor();

  await step("find the annotated run by its tag, then select three runs");
  await nav("Evals").click();
  await filter.fill("tags.team = 'rag'");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await page.getByTestId("search-total").getByText("1 of 3").waitFor();
  await table.getByText("Hybrid with the gateway reranker").waitFor();
  await filter.fill("");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await page.getByTestId("search-total").getByText("3 of 3").waitFor();
  const boxes = page.getByRole("checkbox", { name: /^Select / });
  for (let index = 0; index < 3; index += 1) await boxes.nth(index).check();
  await page.getByRole("button", { name: "Compare 3 selected" }).click();

  await step("compare them: intervals, deltas and groups");
  await rendered("compare-forest").waitFor();
  assert.equal(await page.getByTestId("compare-cell").count(), 12);
  assert.equal(await page.getByTestId("compared-run").count(), 3);
  assert.equal(
    await page
      .locator('[data-testid="compare-cell"] [data-group="leader"]')
      .count(),
    4,
  );
  assert.match(page.url(), /\/compare\/[^/]+\/[^/]+\/[^/]+$/);

  await step(
    "the charts: curves, funnel, where lost, Pareto, parallel coordinates",
  );
  await page.getByRole("tab", { name: "Charts" }).click();
  for (const chart of [
    "compare-curves",
    "compare-funnel",
    "compare-lost",
    "compare-pareto",
    "compare-parallel",
  ]) {
    await rendered(chart).waitFor();
  }

  await step("the parameters that differ, and a query's lists side by side");
  await page.getByRole("tab", { name: "Parameters" }).click();
  await page
    .getByRole("table", { name: "Parameters that differ" })
    .getByRole("cell", { name: "rerank.component", exact: true })
    .waitFor();
  await page.getByRole("tab", { name: "Queries" }).click();
  const queries = page.getByRole("table", { name: "Queries per run" });
  await queries.getByRole("row").nth(1).click();
  await page.getByTestId("side-by-side").nth(2).waitFor();

  await step("latency per step, and the runs' maps with their differences");
  await page.getByRole("tab", { name: "Latency" }).click();
  await rendered("compare-latency").waitFor();
  await rendered("compare-boxes").waitFor();
  await page.getByRole("tab", { name: "Maps" }).click();
  await page.getByTestId("pipeline-map").nth(2).waitFor();
  assert.ok((await page.locator('[data-change="changed"]').count()) >= 1);

  await step("every RAG pipeline, and one's map with its datastores");
  await nav("Maps").click();
  const pipelines = page.getByRole("table", { name: "RAG pipelines" });
  await pipelines.getByRole("row").nth(3).waitFor();
  await pipelines.getByRole("link").first().click();
  await page.getByTestId("pipeline-map").waitFor();
  await page
    .getByTestId("map-summary")
    .getByText(/datastores:/)
    .waitFor();
  await page.locator('[data-node="store:objects"]').waitFor();
  await page.locator('[data-node="chunk"]').click();
  await page.getByTestId("map-node-detail").getByText("chunk/").waitFor();

  await step("models: rerankers by their controlled pair");
  await nav("Models").click();
  await page.getByLabel("Role").selectOption("rerank");
  await rendered("models-bars").waitFor();
  await rendered("models-matrix").waitFor();

  await step("no query text in the console");
  for (const message of console_) {
    assert.ok(
      !/\?$/.test(message.trim()),
      "something like a query reached the console",
    );
  }
  process.stdout.write("ok\n");
} finally {
  await browser.close();
}
