import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { normalizeBasePath, toBrowserPath, toLabPath } from "../src/lab/paths";

describe("base paths", () => {
  it("normalizes to one leading and one trailing slash", () => {
    assert.equal(normalizeBasePath("/stratafi"), "/stratafi/");
    assert.equal(normalizeBasePath("stratafi/"), "/stratafi/");
    assert.equal(normalizeBasePath("//tools/lab//"), "/tools/lab/");
    assert.equal(normalizeBasePath(""), "/");
    assert.equal(normalizeBasePath("/"), "/");
  });

  it("maps lab paths into the base path", () => {
    assert.equal(toBrowserPath("/stratafi/", "/runs/7"), "/stratafi/runs/7");
    assert.equal(toBrowserPath("/tools/lab", "runs"), "/tools/lab/runs");
    assert.equal(toBrowserPath("/stratafi/", "/"), "/stratafi/");
  });

  it("refuses absolute URLs, which would leave the lab", () => {
    assert.throws(() => toBrowserPath("/stratafi/", "https://example.test/x"));
    assert.throws(() => toBrowserPath("/stratafi/", "//example.test/x"));
  });

  it("maps browser paths back, and knows paths outside the lab", () => {
    assert.equal(toLabPath("/stratafi/", "/stratafi/runs/7"), "/runs/7");
    assert.equal(toLabPath("/stratafi/", "/stratafi"), "/");
    assert.equal(toLabPath("/stratafi/", "/stratafi/"), "/");
    assert.equal(toLabPath("/stratafi/", "/stratafix/runs"), null);
    assert.equal(toLabPath("/stratafi/", "/files"), null);
  });

  it("round-trips", () => {
    for (const path of ["/", "/runs", "/runs/7/stages"]) {
      assert.equal(toLabPath("/a/b/", toBrowserPath("/a/b/", path)), path);
    }
  });
});
