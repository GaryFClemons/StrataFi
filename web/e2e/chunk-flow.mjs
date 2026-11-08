// The Chunk step's browser flow: choose the recursive chunker with an overlap,
// run to Chunk, and read the chunk boundaries (overlaps hatched), the token
// histogram with the limit line, the warnings and the chunk list; then
// compare the parent-child chunker side by side.
//
//   node e2e/chunk-flow.mjs <app URL> <folder with food-safety.md>

import assert from "node:assert/strict";
import path from "node:path";
import { assertNoText, corpusFrom, open, signIn, waitForRun } from "./lib.mjs";

const [appUrl, folder] = process.argv.slice(2);
if (!appUrl || !folder) {
  console.error("usage: node e2e/chunk-flow.mjs <app URL> <folder>");
  process.exit(2);
}

const { browser, page, console_, step } = await open();
try {
  await step("sign in as an editor");
  await signIn(page, appUrl);

  await step("upload a document");
  await corpusFrom(
    page,
    "chunking",
    [path.join(folder, "food-safety.md")],
    ["food-safety.md"],
  );

  await step("choose the recursive chunker with an overlap");
  await page.getByRole("link", { name: "Pipelines", exact: true }).click();
  await page.getByRole("link", { name: "4. Chunk" }).click();
  await page
    .getByLabel("Corpus version")
    .selectOption({ label: "chunking v1 (1 documents)" });
  const chunker = page.getByRole("group", { name: "Chunker" });
  await chunker
    .getByLabel("Component", { exact: true })
    .selectOption("chunk/recursive");
  await chunker.getByLabel("size").fill("200");
  await chunker.getByLabel("overlap").fill("40");

  await step("run to Chunk");
  await page.getByLabel("Sample").selectOption("all");
  await page.getByRole("button", { name: "Run to this step" }).click();
  await waitForRun(page);
  await page
    .getByRole("table", { name: "Chunked documents" })
    .getByRole("cell", { name: "chunk/recursive" })
    .waitFor();

  await step("the histogram, the warnings and the bands show");
  await page
    .locator('[data-testid="chart"][data-rendered="true"]')
    .first()
    .waitFor();
  await page.getByLabel("Embedding model's input limit").fill("150");
  const warnings = page.getByRole("list", { name: "Chunk warnings" });
  await warnings.getByText("over limit").first().waitFor();
  const bands = page.getByLabel("Text with chunk boundaries");
  await bands.waitFor();
  assert.ok(
    (await bands.locator('[data-overlap="true"]').count()) > 0,
    "overlaps are hatched",
  );
  const rows = await page
    .getByRole("table", { name: "Chunks", exact: true })
    .getByRole("row")
    .count();
  assert.ok(rows > 3, "the chunk list has the chunks");

  await step("compare the parent-child chunker");
  const compare = page.getByRole("region", { name: "Compare chunkers" });
  await compare.getByLabel("Compare with").selectOption("chunk/parent-child");
  await compare.getByRole("button", { name: "Compare chunkers" }).click();
  await compare
    .getByTestId("compare-chunker")
    .getByText("chunk/parent-child", { exact: true })
    .waitFor({ timeout: 120_000 });
  await compare.getByText(/parent sections/).waitFor();

  await step("no document text in the console");
  assertNoText(console_, ["Food Safety and Allergen", "allergy order"]);
  process.stdout.write("ok\n");
} finally {
  await browser.close();
}
