// The Extract step's browser flow: upload an archive, an email with an
// attachment, a duplicate and a mismatched file; run to Extract; the screen
// shows each document's type against its extension, the mismatch, the
// duplicate group and the archive and attachment tree; a URL to the metadata
// address is refused.
//
//   node e2e/extract-flow.mjs <app URL> <folder with the files>

import assert from "node:assert/strict";
import path from "node:path";
import { assertNoText, corpusFrom, open, runStep, signIn } from "./lib.mjs";

const [appUrl, folder] = process.argv.slice(2);
if (!appUrl || !folder) {
  console.error("usage: node e2e/extract-flow.mjs <app URL> <folder>");
  process.exit(2);
}
const FILES = ["bundle.zip", "notice.eml", "copy.md", "fake.pdf"];
const ENTRIES = [
  "bundle.zip/policies/fees.md",
  "bundle.zip/wires.txt",
  "notice.eml/message.txt",
  "notice.eml/hours.md",
  "copy.md",
  "fake.pdf",
];
const PHRASES = ["Halcyon charges", "tip jar", "cafe hours"];

const { browser, page, console_, step } = await open();
try {
  await step("sign in as an editor");
  await signIn(page, appUrl);

  await step("upload an archive, an email, a duplicate and a mismatched file");
  await corpusFrom(
    page,
    "intake",
    FILES.map((name) => path.join(folder, name)),
    ENTRIES,
  );

  await step("run to Extract");
  await runStep(page, "1. Extract", "intake v1 (6 documents)");
  const table = page.getByRole("table", { name: "Extracted documents" });
  await table.waitFor();
  for (const entry of ENTRIES) {
    await table.getByRole("cell", { name: entry, exact: true }).waitFor();
  }

  await step("the mismatched file is flagged");
  const fake = table.getByRole("row").filter({ hasText: "fake.pdf" });
  await fake.getByText("mismatch").waitFor();
  assert.ok(((await fake.textContent()) ?? "").includes("text/plain"));

  await step("the duplicate group and the tree show");
  const groups = page.getByRole("region", { name: "Duplicate groups" });
  await groups.getByText("bundle.zip/policies/fees.md").waitFor();
  assert.ok(((await groups.textContent()) ?? "").includes("copy.md"));
  const tree = page.getByRole("region", {
    name: "Archive and attachment tree",
  });
  await tree.getByText("2 members").first().waitFor();
  await tree.getByText("hours.md", { exact: true }).waitFor();

  await step("a URL to the metadata address is refused");
  await page.getByLabel("Fetch a URL").fill("http://169.254.169.254/latest/");
  await page.getByRole("button", { name: "Fetch", exact: true }).click();
  await page
    .getByRole("alert")
    .filter({ hasText: /metadata/i })
    .waitFor();

  await step("no document text in the console");
  assertNoText(console_, PHRASES);
  process.stdout.write("ok\n");
} finally {
  await browser.close();
}
