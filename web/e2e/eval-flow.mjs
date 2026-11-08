// The eval results browser flow: open an eval run's results, see its metrics
// with their intervals, drill into a missed query, and see the checkpoint
// where its evidence was lost. No query text may reach the browser console.
//
//   node e2e/eval-flow.mjs <app URL, e.g. http://127.0.0.1:8000/stratafi/> <eval run ID>
//
// services/worker/tests/test_evals_integration.py runs the Halcyon demo corpus
// through an eval run first, serves the built app beside the API, and runs
// this. STRATAFI_SCREENSHOTS=<dir> saves a screenshot after each step.

import assert from "node:assert/strict";
import { createRequire } from "node:module";
import path from "node:path";
import { signIn } from "./lib.mjs";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const [appUrl, evalRunId] = process.argv.slice(2);
if (!appUrl || !evalRunId) {
  console.error("usage: node e2e/eval-flow.mjs <app URL> <eval run ID>");
  process.exit(2);
}

const browser = await chromium.launch(
  process.env.STRATAFI_CHROMIUM
    ? { executablePath: process.env.STRATAFI_CHROMIUM }
    : {},
);
const console_ = [];
try {
  const page = await browser.newPage();
  page.setDefaultTimeout(30_000);
  page.on("console", (message) => console_.push(message.text()));
  page.on("pageerror", (error) => console_.push(String(error)));

  let shots = 0;
  const step = async (name) => {
    if (process.env.STRATAFI_SCREENSHOTS && shots > 0) {
      await page.screenshot({
        path: path.join(process.env.STRATAFI_SCREENSHOTS, `eval-${shots}.png`),
        fullPage: true,
      });
    }
    shots += 1;
    process.stdout.write(`- ${name}\n`);
  };

  await step("sign in as an editor");
  await signIn(page, appUrl);

  await step("open the eval run's results");
  await page.getByRole("link", { name: "Evals", exact: true }).click();
  await page
    .getByRole("table", { name: "Eval runs" })
    .getByRole("link", { name: evalRunId.slice(0, 8) })
    .click();
  const cards = page.getByTestId("metric-card");
  await cards.first().waitFor();
  assert.equal(await cards.count(), 4);
  for (const text of await page.getByTestId("metric-ci").allTextContents()) {
    assert.match(text, /^95% CI \d\.\d{3}–\d\.\d{3}, n = \d+$/);
  }
  await page.getByRole("table", { name: "Checkpoints" }).waitFor();

  await step("drill into a missed query");
  await page.getByLabel("Show queries").selectOption("miss");
  const queries = page.getByRole("table", { name: "Queries" });
  const firstMiss = queries.getByRole("link").first();
  await firstMiss.waitFor();
  await firstMiss.click();

  await step("see the checkpoint where its evidence was lost");
  const lost = page.getByTestId("lost-at");
  await lost.waitFor();
  const verdict = (await lost.textContent()) ?? "";
  assert.match(verdict, /^Missed\. (Parse|Retrieval|Rerank):/);
  assert.equal(
    await page.getByTestId("found-in-final").getAttribute("data-found"),
    "no",
  );
  const stage = verdict.match(/^Missed\. (\w+):/)?.[1];
  if (stage === "Retrieval") {
    assert.equal(
      await page.getByTestId("found-in-candidates").getAttribute("data-found"),
      "no",
    );
  } else if (stage === "Rerank") {
    assert.equal(
      await page.getByTestId("found-in-candidates").getAttribute("data-found"),
      "yes",
    );
  }
  await page.getByRole("button", { name: /^Candidates/ }).click();
  await page.getByTestId("eval-passage").first().waitFor();

  await step("no query text in the console");
  const query = (await page.getByLabel("Query").textContent()) ?? "";
  assert.ok(query.length > 10, "the query is shown");
  for (const message of console_) {
    assert.ok(!message.includes(query), "a query reached the console");
  }
  process.stdout.write("ok\n");
} finally {
  await browser.close();
}
