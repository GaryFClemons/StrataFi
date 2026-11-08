// The Tests screen's browser flow: write a test suite in the editor (a problem
// shown as it's typed, a completion picked), save and run it, follow the run to
// its result, open a check's evidence and download its JUnit report; then see
// the suite's runs, trend and automation, make and revoke an API token, add
// and remove a webhook, and see custom checks off until the sandbox's review
// is signed off. The suite's golden query must not reach the browser console.
//
//   node e2e/tests-flow.mjs <app URL>
//
// services/worker/tests/test_testruns_integration.py prepares the Halcyon demo corpus's
// corpus, pipeline spec, retrieval config and eval set first, serves the built
// app beside the API, and runs this. STRATAFI_SCREENSHOTS=<dir> saves a
// screenshot after each step.

import assert from "node:assert/strict";
import { RUN_TIMEOUT, open, signIn } from "./lib.mjs";

const [appUrl] = process.argv.slice(2);
if (!appUrl) {
  console.error("usage: node e2e/tests-flow.mjs <app URL>");
  process.exit(2);
}

const GOLDEN =
  "flow golden query about how fast hourly staff accrue vacation time";
const SUITE = `schema: stratafi.test-suite.v1
name: browser-guard
target:
  pipeline: baseline-v1
  retrieval: hybrid-v1
corpus: halcyon@latest
eval_set: {name: halcyon-retrieval}
checks:
  - id: recall-at-5
    metric: recall@5
    min: 0.0
  - id: everything-parsed
    stage: parse
    metric: parse.success_rate
    min: 1.0
  - id: pto-golden
    query: ${GOLDEN}
    asker: staff
    expect: {in_top_k: 10, doc_key: pto-leave}
`;

const { browser, page, console_, step } = await open();
// The standalone host confirms with the browser's own dialog.
page.on("dialog", (dialog) => void dialog.accept());
try {
  await step("sign in as an owner and open Tests");
  await signIn(page, appUrl, "owner");
  await page.getByRole("link", { name: "Tests", exact: true }).click();
  await page.getByText("Test suites").first().waitFor();
  const editor = page.getByLabel("Suite (YAML, stratafi.test-suite.v1)");

  await step("a problem shows as the suite is typed");
  await editor.fill(SUITE.replace("    min: 0.0\n", ""));
  const problems = page.getByRole("list", { name: "Problems" });
  await problems.getByText("threshold missing").waitFor();

  await step("a completion writes a key");
  await editor.fill("sc");
  await editor.press("End");
  const completions = page.getByRole("group", { name: "Completions" });
  await completions.getByRole("button", { name: "schema" }).click();
  assert.equal(await editor.inputValue(), "schema: ");

  await step("save and run the suite");
  await editor.fill(SUITE);
  await page.getByTestId("suite-valid").waitFor();
  await page.getByRole("button", { name: "Save and run" }).click();
  const status = page.getByTestId("test-run-status");
  await status
    .locator('[data-status="passed"]')
    .waitFor({ timeout: RUN_TIMEOUT });
  const checks = page.getByRole("table", { name: "Checks" });
  for (const id of ["recall-at-5", "everything-parsed", "pto-golden"]) {
    await checks.getByText(id, { exact: true }).waitFor();
  }
  assert.equal(await checks.locator('[data-status="pass"]').count(), 3);

  await step("download the JUnit report");
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: "JUnit XML" }).click(),
  ]);
  const junit = await (await download.createReadStream()).toArray();
  const xml = Buffer.concat(junit).toString("utf8");
  assert.match(xml, /<testsuites name="stratafi" tests="3" failures="0"/);
  assert.ok(!xml.includes(GOLDEN), "the report names checks, not queries");

  await step("open a check's evidence");
  await checks
    .getByRole("link", { name: /^eval_run / })
    .first()
    .click();
  await page.getByRole("link", { name: "Evals", exact: true }).waitFor();
  assert.match(page.url(), /\/evals\/[0-9a-f-]{36}$/);
  await page.goBack();

  await step("the suite: its runs, trend and automation");
  await page.getByRole("link", { name: "The suite" }).click();
  await page.getByRole("table", { name: "Runs of this suite" }).waitFor();
  await page.getByRole("button", { name: "Run now" }).click();
  await page
    .getByTestId("test-run-status")
    .locator('[data-status="passed"]')
    .waitFor({ timeout: RUN_TIMEOUT });
  await page.getByRole("link", { name: "The suite" }).click();
  const runs = page.getByRole("table", { name: "Runs of this suite" });
  await runs.locator('[data-status="passed"]').nth(1).waitFor();
  await page
    .locator('[data-testid="check-trend"][data-rendered="true"]')
    .waitFor();
  await page.getByTestId("suite-automation").waitFor();
  await page.getByRole("link", { name: "Tests", exact: true }).click();
  await page.getByTestId("suite-trend").first().waitFor();

  await step("make an API token and revoke it");
  await page.getByRole("link", { name: /Automation/ }).click();
  // Each suite's schedules load above the forms and move them down: let the
  // page settle first, or a click can land on whatever moved under it.
  await page.getByTestId("suite-automation").first().waitFor();
  await page.getByTestId("custom-checks").waitFor();
  await page.getByLabel("Token name").fill("ci-browser");
  await page.getByRole("button", { name: "Make token" }).click();
  const token = page.getByTestId("new-token").locator("code");
  assert.match(await token.innerText(), /^sft_/);
  const tokens = page.getByRole("table", { name: "API tokens" });
  await tokens.getByText("ci-browser").waitFor();
  await tokens.getByRole("button", { name: "Revoke" }).click();
  await tokens.getByText("revoked or expired").waitFor();

  await step("add a webhook and remove it");
  await page.getByLabel("Webhook URL").fill("http://127.0.0.1:9/hook");
  await page.getByRole("button", { name: "Add webhook" }).click();
  const webhooks = page.getByRole("table", { name: "Webhooks" });
  await webhooks.getByText("http://127.0.0.1:9/hook").waitFor();
  await webhooks.getByRole("button", { name: "Remove" }).click();
  await page.getByText("No webhooks yet.").waitFor();

  await step("custom checks stay off until the sandbox's review");
  const custom = page.getByTestId("custom-checks");
  await custom.getByText("off", { exact: true }).waitFor();
  await custom.getByText(/no custom-check sandbox/).waitFor();
  assert.ok(
    await custom
      .getByRole("button", { name: "Turn custom checks on" })
      .isDisabled(),
    "custom checks can't be turned on without the sandbox",
  );

  await step("the golden query never reached the console");
  assert.ok(
    console_.every((line) => !line.includes(GOLDEN)),
    "a golden query reached the console",
  );
  console.log("ok");
} finally {
  await browser.close();
}
