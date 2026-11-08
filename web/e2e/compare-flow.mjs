// The Compare browser flow: compare two eval runs and see each metric's
// delta with its interval. No query text may reach the browser console.
//
//   node e2e/compare-flow.mjs <app URL> <eval run A> <eval run B>
//
// services/worker/tests/test_compare_integration.py runs two retrieval
// configs over the Halcyon demo corpus first, serves the built app beside the API,
// and runs this.

import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { signIn } from "./lib.mjs";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const [appUrl, first, second] = process.argv.slice(2);
if (!appUrl || !first || !second) {
  console.error(
    "usage: node e2e/compare-flow.mjs <app URL> <eval run A> <eval run B>",
  );
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

  process.stdout.write("- sign in as an editor\n");
  await signIn(page, appUrl);

  process.stdout.write("- choose two eval runs to compare\n");
  await page.getByRole("link", { name: "Evals", exact: true }).click();
  await page.getByRole("link", { name: "Compare two runs" }).click();
  const pick = async (label, id) => {
    const select = page.getByLabel(label, { exact: true });
    const option = select.locator(`option[value="${id}"]`);
    await option.waitFor({ state: "attached" });
    await select.selectOption(id);
  };
  await pick("Eval run A", first);
  await pick("Eval run B", second);

  process.stdout.write("- see each delta with its interval\n");
  const deltas = page.getByTestId("compare-delta");
  await deltas.first().waitFor();
  assert.equal(await deltas.count(), 4);
  for (const text of await deltas.allTextContents()) {
    assert.match(text, /^[+-]?\d\.\d{3} \(-?\d\.\d{3}–-?\d\.\d{3}\)$/);
  }
  const table = page.getByRole("table", { name: "Deltas" });
  await table.getByRole("cell", { name: "recall@5", exact: true }).waitFor();
  await page
    .getByRole("table", { name: "Retrieval config" })
    .getByRole("cell", { name: "rerank.component" })
    .waitFor();
  assert.ok(page.url().endsWith(`/evals/compare/${first}/${second}`));

  process.stdout.write("- no query text in the console\n");
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
