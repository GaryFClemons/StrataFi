// The cleaning browser flow: add the redaction rule on the Scan + Clean
// screen, set a policy, run to the step, and read the cleaning diff: the
// redaction shows its token, and the value appears nowhere on the page or in
// the console.
//
//   node e2e/clean-flow.mjs <app URL> <folder with records.md>

import assert from "node:assert/strict";
import path from "node:path";
import { assertNoText, corpusFrom, open, signIn, waitForRun } from "./lib.mjs";

const [appUrl, folder] = process.argv.slice(2);
if (!appUrl || !folder) {
  console.error("usage: node e2e/clean-flow.mjs <app URL> <folder>");
  process.exit(2);
}
const SSN = "123-45-6789";

const { browser, page, console_, step } = await open();
try {
  await step("sign in as an editor");
  await signIn(page, appUrl);

  await step("upload a document with personal data");
  await corpusFrom(
    page,
    "cleaning",
    [path.join(folder, "records.md")],
    ["records.md"],
  );

  await step("add the redaction rule and set a policy");
  await page.getByRole("link", { name: "Pipelines", exact: true }).click();
  await page.getByRole("link", { name: "2. Scan + Clean" }).click();
  await page
    .getByLabel("Corpus version")
    .selectOption({ label: "cleaning v1 (1 documents)" });
  await page.getByLabel("Policy for duplicate").selectOption("flag");
  await page
    .getByLabel("Add a rule")
    .selectOption({ label: "Redact personal data" });
  await page.getByRole("button", { name: "Add rule" }).click();
  await page.getByRole("group", { name: "Rule 2" }).waitFor();
  await page.getByRole("button", { name: "Move rule 2 up" }).click();
  const first = page
    .getByRole("group", { name: "Rule 1" })
    .getByLabel("Component", { exact: true });
  assert.equal(await first.inputValue(), "clean/redact");

  await step("run to Scan + Clean");
  await page.getByLabel("Sample").selectOption("all");
  await page.getByRole("button", { name: "Run to this step" }).click();
  await waitForRun(page);

  await step("the diff shows the redaction without the value");
  const cleaning = page.getByRole("table", { name: "Cleaning" });
  await cleaning
    .getByRole("cell", { name: "records.md", exact: true })
    .waitFor();
  const diff = page.getByRole("region", { name: "Cleaning diff" });
  const redaction = diff
    .getByTestId("diff-hunk")
    .filter({ hasText: "redaction" })
    .first();
  await redaction.waitFor();
  assert.ok(
    ((await redaction.getByLabel("After").textContent()) ?? "").includes(
      "[SSN]",
    ),
  );
  const content = (await page.content()) ?? "";
  assert.ok(!content.includes(SSN), "the redacted value reached the page");
  assert.ok(
    !content.includes(SSN.slice(0, 7)),
    "part of the value reached the page",
  );

  await step("no document text in the console");
  assertNoText(console_, [SSN, "test record"]);
  process.stdout.write("ok\n");
} finally {
  await browser.close();
}
