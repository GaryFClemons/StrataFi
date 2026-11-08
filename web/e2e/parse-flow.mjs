// The Parse step's browser flow: route PDFs to pdfplumber and CSV to the
// spreadsheet parser, run to Parse, see the original page rendered by pdf.js
// with element boxes over it, read a table in the table viewer, compare a
// second parser on the same page, and author an eval item from a passage.
//
//   node e2e/parse-flow.mjs <app URL> <folder with wholesale.pdf, tiers.csv>

import assert from "node:assert/strict";
import path from "node:path";
import { assertNoText, corpusFrom, open, signIn, waitForRun } from "./lib.mjs";

const [appUrl, folder] = process.argv.slice(2);
if (!appUrl || !folder) {
  console.error("usage: node e2e/parse-flow.mjs <app URL> <folder>");
  process.exit(2);
}

const { browser, page, console_, step } = await open();
try {
  await step("sign in as an editor");
  await signIn(page, appUrl);

  await step("upload a PDF and a CSV");
  await corpusFrom(
    page,
    "parsing",
    ["wholesale.pdf", "tiers.csv"].map((name) => path.join(folder, name)),
    ["wholesale.pdf", "tiers.csv"],
  );

  await step("route PDFs to pdfplumber and CSV to the spreadsheet parser");
  await page.getByRole("link", { name: "Pipelines", exact: true }).click();
  await page.getByRole("link", { name: "3. Parse" }).click();
  await page
    .getByLabel("Corpus version")
    .selectOption({ label: "parsing v1 (2 documents)" });
  await page.getByLabel("Add a route").fill("pdf");
  await page.getByRole("button", { name: "Add route" }).click();
  const pdfRoute = page.getByRole("group", { name: "Route pdf" });
  await pdfRoute
    .getByLabel("Component", { exact: true })
    .selectOption("parse/pdfplumber");
  await page.getByLabel("Add a route").fill("csv");
  await page.getByRole("button", { name: "Add route" }).click();
  const csvRoute = page.getByRole("group", { name: "Route csv" });
  await csvRoute
    .getByLabel("Component", { exact: true })
    .selectOption("parse/spreadsheet");
  await csvRoute.getByLabel("layout").selectOption("table");

  await step("run to Parse");
  await page.getByLabel("Sample").selectOption("all");
  await page.getByRole("button", { name: "Run to this step" }).click();
  await waitForRun(page);
  const parsed = page.getByRole("table", { name: "Parsed documents" });
  await parsed.getByRole("cell", { name: "parse/pdfplumber" }).waitFor();

  await step("the page renders with element boxes");
  await parsed.getByRole("cell", { name: "wholesale.pdf" }).click();
  await page
    .locator('[data-testid="pdf-page"][data-rendered="true"]')
    .waitFor();
  const boxes = await page.getByTestId("element-box").count();
  assert.ok(boxes > 0, "element boxes are drawn over the page");
  await page.getByTestId("element-box").first().click();
  await page.getByRole("list", { name: "Parsed elements" }).waitFor();

  await step("compare pypdf on the same page");
  const compare = page.getByRole("region", { name: "Compare parsers" });
  await compare.getByLabel("pypdf text layer").check();
  await compare.getByRole("button", { name: "Compare" }).click();
  await compare
    .getByTestId("compare-column")
    .getByText("parse/pypdf", { exact: true })
    .waitFor({ timeout: 120_000 });

  await step("ask a question about a passage");
  await page
    .getByRole("list", { name: "Parsed elements" })
    .getByRole("button", { name: "Ask a question about this passage" })
    .first()
    .click();
  await page
    .getByRole("form", { name: "Ask a question about this passage" })
    .getByLabel("Question")
    .fill("How are wholesale price tiers set?");
  await page.getByRole("button", { name: "Add to eval set" }).click();
  await page.getByText(/Added authored-1 to authored v1/).waitFor();

  await step("the table viewer shows the CSV's cells");
  await parsed.getByRole("cell", { name: "tiers.csv" }).click();
  const viewer = page.getByLabel("Table viewer").first();
  await viewer.waitFor();
  await viewer.getByRole("columnheader", { name: "Tier" }).waitFor();
  await viewer.getByRole("cell", { name: "9.10" }).waitFor();

  await step("no document text in the console");
  assertNoText(console_, [
    "Wholesale Pricing",
    "Tier 3: Preferred",
    "Flagship",
  ]);
  process.stdout.write("ok\n");
} finally {
  await browser.close();
}
