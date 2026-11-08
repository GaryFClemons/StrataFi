import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  coerce,
  defaultsFromSchema,
  display,
  fieldsFromSchema,
  labelFor,
  type Field,
  type JsonSchema,
} from "../src/lab/spec/schemaForm";

// Parameter schemas as GET /components returns them (Pydantic's JSON Schema).
const HEADING_PACK = {
  additionalProperties: false,
  properties: {
    max_tokens: {
      default: 800,
      maximum: 32000,
      minimum: 16,
      title: "Max Tokens",
      type: "integer",
    },
    overlap: {
      default: 0.1,
      description: "Of max_tokens, when a section splits",
      exclusiveMaximum: 1,
      minimum: 0,
      title: "Overlap",
      type: "number",
    },
    token_estimator: {
      const: "chars/4",
      default: "chars/4",
      title: "Token Estimator",
      type: "string",
    },
  },
  title: "HeadingPackParams",
  type: "object",
};
const GATEWAY = {
  properties: {
    normalize: { default: false, title: "Normalize", type: "boolean" },
    document_prefix: {
      default: "",
      maxLength: 64,
      title: "Document Prefix",
      type: "string",
    },
    alias: {
      default: "embed-default",
      pattern: "^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$",
      title: "Alias",
      type: "string",
    },
  },
  type: "object",
};
const WHITESPACE = {
  properties: {
    paragraphs: {
      default: "keep",
      enum: ["keep", "trim", "collapse"],
      type: "string",
    },
  },
  type: "object",
};
const NULLABLE = {
  properties: {
    limit: {
      anyOf: [{ type: "integer", minimum: 1 }, { type: "null" }],
      default: null,
    },
    nested: { $ref: "#/$defs/Nested" },
  },
  type: "object",
};

function field(schema: JsonSchema, name: string): Field {
  const found = fieldsFromSchema(schema).find((item) => item.name === name);
  assert.ok(found, name);
  return found;
}

describe("fields from a schema", () => {
  it("keep the properties' order, labels and help", () => {
    const fields = fieldsFromSchema(HEADING_PACK);
    assert.deepEqual(
      fields.map((item) => [item.name, item.label, item.kind]),
      [
        ["max_tokens", "Max tokens", "integer"],
        ["overlap", "Overlap", "number"],
        ["token_estimator", "Token estimator", "choice"],
      ],
    );
    assert.equal(
      fields[1]?.description,
      "Of max_tokens, when a section splits",
    );
    assert.equal(fields[1]?.exclusiveMaximum, 1);
  });

  it("turn enums and constants into choices", () => {
    assert.deepEqual(field(WHITESPACE, "paragraphs").options, [
      "keep",
      "trim",
      "collapse",
    ]);
    assert.deepEqual(field(HEADING_PACK, "token_estimator").options, [
      "chars/4",
    ]);
  });

  it("know nullable fields and fall back to JSON for nested models", () => {
    const limit = field(NULLABLE, "limit");
    assert.equal(limit.kind, "integer");
    assert.equal(limit.nullable, true);
    assert.equal(limit.minimum, 1);
    assert.equal(field(NULLABLE, "nested").kind, "json");
  });

  it("collect defaults: the baseline's parameters", () => {
    assert.deepEqual(defaultsFromSchema(HEADING_PACK), {
      max_tokens: 800,
      overlap: 0.1,
      token_estimator: "chars/4",
    });
    assert.deepEqual(defaultsFromSchema({ properties: {} }), {});
    assert.deepEqual(defaultsFromSchema({}), {});
  });

  it("label snake_case names", () => {
    assert.equal(
      labelFor("max_signature_age_hours"),
      "Max signature age hours",
    );
    assert.equal(labelFor("k1"), "K1");
  });
});

describe("checking input", () => {
  it("parses whole numbers within bounds", () => {
    const tokens = field(HEADING_PACK, "max_tokens");
    assert.deepEqual(coerce(tokens, " 400 "), { ok: true, value: 400 });
    assert.deepEqual(coerce(tokens, "8"), { ok: false, error: "At least 16" });
    assert.deepEqual(coerce(tokens, "40000"), {
      ok: false,
      error: "At most 32000",
    });
    assert.deepEqual(coerce(tokens, "4.5"), {
      ok: false,
      error: "A whole number",
    });
    assert.deepEqual(coerce(tokens, ""), { ok: true, value: 800 });
  });

  it("parses numbers with exclusive bounds", () => {
    const overlap = field(HEADING_PACK, "overlap");
    assert.deepEqual(coerce(overlap, "0.25"), { ok: true, value: 0.25 });
    assert.deepEqual(coerce(overlap, "1"), { ok: false, error: "Less than 1" });
    assert.deepEqual(coerce(overlap, "abc"), { ok: false, error: "A number" });
  });

  it("checks text against its length and pattern", () => {
    const alias = field(GATEWAY, "alias");
    assert.deepEqual(coerce(alias, "embed-local"), {
      ok: true,
      value: "embed-local",
    });
    assert.equal(coerce(alias, "bad alias!").ok, false);
    const prefix = field(GATEWAY, "document_prefix");
    assert.deepEqual(coerce(prefix, ""), { ok: true, value: "" });
    assert.equal(coerce(prefix, "x".repeat(65)).ok, false);
  });

  it("takes only the offered choices, and booleans", () => {
    const paragraphs = field(WHITESPACE, "paragraphs");
    assert.deepEqual(coerce(paragraphs, "trim"), { ok: true, value: "trim" });
    assert.equal(coerce(paragraphs, "squash").ok, false);
    const normalize = field(GATEWAY, "normalize");
    assert.deepEqual(coerce(normalize, "true"), { ok: true, value: true });
    assert.equal(coerce(normalize, "yes").ok, false);
  });

  it("empties a nullable field to null", () => {
    assert.deepEqual(coerce(field(NULLABLE, "limit"), ""), {
      ok: true,
      value: null,
    });
  });

  it("parses JSON for fields it can't model", () => {
    const nested = field(NULLABLE, "nested");
    assert.deepEqual(coerce(nested, '{"a": 1}'), { ok: true, value: { a: 1 } });
    assert.equal(coerce(nested, "{").ok, false);
    assert.equal(display(nested, { a: 1 }), '{"a":1}');
  });
});
