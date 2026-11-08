// The pipeline browser flow: sign in, upload three fixture documents, run to
// Parse on a sample and read the parsed text, run to Embed, and find a fixture
// passage with search. No document text may reach the browser console.
//
//   node e2e/pipeline-flow.mjs <app URL, e.g. http://127.0.0.1:8000/stratafi/> <fixtures dir>
//
// STRATAFI_SCREENSHOTS=<dir> saves a screenshot after each step.
//
// services/worker/tests/test_runs_integration.py starts the API, a worker and
// the fakes, serves the built app, and runs this. It uses the `playwright`
// library from wherever Node resolves it (the machine's install, NODE_PATH).

import assert from "node:assert/strict";
import { createRequire } from "node:module";
import path from "node:path";
import { signIn } from "./lib.mjs";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const [appUrl, fixtures] = process.argv.slice(2);
if (!appUrl || !fixtures) {
  console.error("usage: node e2e/pipeline-flow.mjs <app URL> <fixtures dir>");
  process.exit(2);
}
const DOCUMENTS = ["tips.pdf", "wholesale.txt", "food-safety.docx"];
const QUERY = "Who counts the tip jar at close?";
// Phrases from the fixtures: the parsed text shows one, the console none.
const PHRASES = ["tip jar", "Tip Policy", "Wholesale Pricing", "allergen"];
const RUN_TIMEOUT = 120_000;

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
        path: path.join(process.env.STRATAFI_SCREENSHOTS, `${shots}.png`),
        fullPage: true,
      });
    }
    shots += 1;
    process.stdout.write(`- ${name}\n`);
  };
  const waitForRun = async (status) => {
    await page
      .locator(`[data-testid="run-progress"] [data-status="${status}"]`)
      .waitFor({ timeout: RUN_TIMEOUT });
  };

  await step("sign in as an editor");
  await signIn(page, appUrl);

  await step("upload three fixture documents and freeze a version");
  await page.getByRole("link", { name: "Documents", exact: true }).click();
  await page.getByLabel("New corpus").fill("fixtures");
  await page.getByRole("button", { name: "Create corpus" }).click();
  await page.getByText("Corpus fixtures").waitFor();
  await page
    .getByLabel("Upload documents")
    .setInputFiles(DOCUMENTS.map((name) => path.join(fixtures, name)));
  await page.getByRole("button", { name: "Upload", exact: true }).click();
  const documents = page.getByRole("table", { name: "Documents" });
  for (const name of DOCUMENTS) {
    await documents.getByRole("cell", { name, exact: true }).waitFor();
  }
  await page.getByRole("button", { name: "Freeze version" }).click();
  await page
    .getByRole("table", { name: "Versions" })
    .getByRole("cell", { name: "v1", exact: true })
    .waitFor();

  await step("run to Parse on a sample and read the parsed text");
  await page.getByRole("link", { name: "Pipelines", exact: true }).click();
  await page.getByRole("link", { name: "3. Parse" }).click();
  await page
    .getByLabel("Corpus version")
    .selectOption({ label: "fixtures v1 (3 documents)" });
  await page.getByLabel("Sample").selectOption("5");
  await page.getByRole("button", { name: "Run to this step" }).click();
  await waitForRun("succeeded");
  const parsed = page.getByRole("list", { name: "Parsed elements" });
  await parsed.getByTestId("parsed-element").first().waitFor();
  const text = (await parsed.textContent()) ?? "";
  assert.ok(
    PHRASES.some((phrase) => text.includes(phrase)),
    "the parsed text shows a fixture's text",
  );

  await step("run to Embed on every document");
  await page.getByRole("link", { name: "5. Embed" }).click();
  await page.getByLabel("Sample").selectOption("all");
  await page.getByRole("button", { name: "Run to this step" }).click();
  await waitForRun("succeeded");
  const embeddings = page.getByRole("table", { name: "Embeddings" });
  await embeddings.waitFor();
  for (const name of DOCUMENTS) {
    await embeddings.getByRole("cell", { name, exact: true }).waitFor();
  }
  await page.getByRole("table", { name: "Indexes" }).waitFor();

  await step("search finds a fixture passage");
  await page.getByRole("link", { name: "6. Search" }).click();
  await page.getByLabel("Query").fill(QUERY);
  await page.getByRole("button", { name: "Search", exact: true }).click();
  const first = page.getByTestId("search-hit").first();
  await first.waitFor();
  assert.equal(await first.getByTestId("hit-entry").textContent(), "tips.pdf");
  assert.ok(
    ((await first.textContent()) ?? "")
      .replace(/\s+/g, " ")
      .includes("tip jar"),
  );

  await step("no document text in the console");
  for (const message of console_) {
    for (const phrase of PHRASES) {
      assert.ok(!message.includes(phrase), "document text reached the console");
    }
  }
  process.stdout.write("ok\n");
} finally {
  await browser.close();
}
