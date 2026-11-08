// The Embed step's browser flow: pick a model from the catalog (the gateway's
// alias), set Matryoshka truncation, int8 storage and an input limit, run to
// Embed with the exact index, read the embeddings (dimensions, dtype,
// truncated inputs), the indexes, the projection, and a chunk's nearest
// neighbours; then switch the dense index to pgvector HNSW and see its ANN
// recall.
//
//   node e2e/embed-flow.mjs <app URL> <fixtures folder>

import assert from "node:assert/strict";
import path from "node:path";
import {
  RUN_TIMEOUT,
  assertNoText,
  corpusFrom,
  open,
  signIn,
  waitForRun,
} from "./lib.mjs";

const [appUrl, folder] = process.argv.slice(2);
if (!appUrl || !folder) {
  console.error("usage: node e2e/embed-flow.mjs <app URL> <folder>");
  process.exit(2);
}
const FILES = ["food-safety.md", "tips.txt"];

const { browser, page, console_, step } = await open();
try {
  await step("sign in as an editor");
  await signIn(page, appUrl);
  await corpusFrom(
    page,
    "embedding",
    FILES.map((name) => path.join(folder, name)),
    FILES,
  );

  await step("choose a model and its options");
  await page.getByRole("link", { name: "Pipelines", exact: true }).click();
  await page.getByRole("link", { name: "5. Embed" }).click();
  await page
    .getByLabel("Corpus version")
    .selectOption({ label: "embedding v1 (2 documents)" });
  const catalog = page.getByRole("table", { name: "Embedding models" });
  await catalog.getByRole("cell", { name: "BGE small v1.5" }).waitFor();
  await catalog.getByRole("button", { name: "Use embed-local" }).click();
  const embedder = page.getByRole("group", { name: "Embedder" });
  await embedder.getByLabel("normalize").check();
  await embedder.getByLabel("Dimensions", { exact: true }).fill("256");
  await embedder.getByLabel("dtype").selectOption("int8");
  await embedder.getByLabel("Max input tokens").fill("100");

  await step("run to Embed");
  await page.getByLabel("Sample").selectOption("all");
  await page.getByRole("button", { name: "Run to this step" }).click();
  await waitForRun(page);
  const embeddings = page.getByRole("table", { name: "Embeddings" });
  await embeddings.getByRole("cell", { name: "256 of 768" }).first().waitFor();
  await embeddings.getByRole("cell", { name: "int8" }).first().waitFor();
  await embeddings
    .getByText(/over the limit/)
    .first()
    .waitFor();
  await page
    .getByRole("table", { name: "Indexes" })
    .getByRole("cell", { name: "exact" })
    .first()
    .waitFor();

  await step("the projection and the nearest neighbours");
  await page.locator('[data-testid="chart"][data-rendered="true"]').waitFor();
  const explore = page.getByLabel("Explore a chunk");
  const options = await explore.locator("option").count();
  assert.ok(options > 2, "the projection lists chunks");
  await explore.selectOption({ index: 1 });
  await page.getByTestId("neighbour").first().waitFor();

  await step("switch the dense index to pgvector HNSW");
  await page
    .getByRole("group", { name: "Dense index" })
    .getByLabel("Component", { exact: true })
    .selectOption("index/pgvector-hnsw");
  await page.getByRole("button", { name: "Run to this step" }).click();
  await waitForRun(page);
  const indexes = page.getByRole("table", { name: "Indexes" });
  const hnsw = indexes
    .getByRole("row")
    .filter({ hasText: "index/pgvector-hnsw" });
  // The run panel can still show the previous run's status, so wait for this
  // index's own task: its recall is measured when it's built.
  await hnsw
    .locator('[data-status="succeeded"]')
    .waitFor({ timeout: RUN_TIMEOUT });
  const recall = await hnsw.getByRole("cell").nth(4).textContent();
  assert.ok(Number(recall) > 0 && Number(recall) <= 1, `ANN recall ${recall}`);

  await step("no document text in the console");
  assertNoText(console_, ["Food Safety and Allergen", "Tip Policy"]);
  process.stdout.write("ok\n");
} finally {
  await browser.close();
}
