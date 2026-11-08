// The search playground's browser flow: run three documents to Embed, pick an eval
// item in the search playground (query, asker and evidence come with it),
// compare config A (hybrid RRF, no reranker) with config B (keyword only,
// the word-overlap reranker), and read both sides' checkpoints with the
// evidence highlighted.
//
//   node e2e/playground-flow.mjs <app URL> <fixtures folder>

import assert from "node:assert/strict";
import path from "node:path";
import { assertNoText, corpusFrom, open, signIn, waitForRun } from "./lib.mjs";

const [appUrl, folder] = process.argv.slice(2);
if (!appUrl || !folder) {
  console.error("usage: node e2e/playground-flow.mjs <app URL> <folder>");
  process.exit(2);
}
const FILES = ["tips.pdf", "wholesale.txt", "food-safety.docx"];

const { browser, page, console_, step } = await open();
try {
  await step("sign in, upload and run to Embed");
  await signIn(page, appUrl);
  await corpusFrom(
    page,
    "playground",
    FILES.map((name) => path.join(folder, name)),
    FILES,
  );
  await page.getByRole("link", { name: "Pipelines", exact: true }).click();
  await page.getByRole("link", { name: "5. Embed" }).click();
  await page
    .getByLabel("Corpus version")
    .selectOption({ label: "playground v1 (3 documents)" });
  await page.getByLabel("Sample").selectOption("all");
  await page.getByRole("button", { name: "Run to this step" }).click();
  await waitForRun(page);

  await step("pick an eval item in the playground");
  await page.getByRole("link", { name: "6. Search" }).click();
  await page.getByLabel("Eval set").selectOption({ label: "tip-questions" });
  await page.getByLabel("Item").selectOption("tip-jar-count");
  const query = await page.getByLabel("Query").inputValue();
  assert.ok(query.includes("tip jar"), "the item's query fills the form");
  await page.getByText("Evidence from the eval item is highlighted.").waitFor();

  await step("compare two configs");
  await page.getByLabel("Compare two configs").check();
  const configB = page.getByRole("group", { name: "Config B" });
  await configB.getByLabel("Dense").uncheck();
  await configB.getByLabel("Reranker").selectOption("rerank/fake-overlap");
  await page.getByRole("button", { name: "Search", exact: true }).click();

  await step("both sides show their checkpoints and the evidence");
  const resultsA = page.getByRole("region", { name: "Results A" });
  const resultsB = page.getByRole("region", { name: "Results B" });
  await resultsA.getByTestId("search-hit").first().waitFor();
  await resultsB.getByTestId("search-hit").first().waitFor();
  assert.equal(
    await resultsA.getByTestId("hit-entry").first().textContent(),
    "tips.pdf",
  );
  await resultsA.getByTestId("evidence").first().waitFor();
  const tabsA = page.getByRole("tablist", { name: "Checkpoints A" });
  await tabsA.getByRole("tab", { name: "Dense" }).waitFor();
  await tabsA.getByRole("tab", { name: "Fused" }).click();
  await resultsA.getByTestId("search-hit").first().waitFor();
  const tabsB = page.getByRole("tablist", { name: "Checkpoints B" });
  assert.equal(await tabsB.getByRole("tab", { name: "Dense" }).count(), 0);
  await tabsB.getByRole("tab", { name: "Reranked" }).click();
  await resultsB.getByTestId("search-hit").first().waitFor();

  await step("no document text in the console");
  assertNoText(console_, ["count the tip jar", "Wholesale Pricing"]);
  process.stdout.write("ok\n");
} finally {
  await browser.close();
}
