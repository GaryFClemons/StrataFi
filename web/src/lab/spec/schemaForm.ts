// Forms from component parameter schemas: the API returns each
// component's JSON Schema (Pydantic's), and these pure functions turn it into
// fields, defaults and checked values. A new component needs no form code.

export type JsonSchema = Record<string, unknown>;

export type FieldKind =
  "integer" | "number" | "boolean" | "choice" | "text" | "json";

export interface Field {
  name: string;
  label: string;
  kind: FieldKind;
  description?: string;
  default?: unknown;
  nullable: boolean;
  /** Choices for "choice" fields; one choice for a constant. */
  options?: (string | number)[];
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  maxLength?: number;
  pattern?: string;
}

export type Coerced =
  { ok: true; value: unknown } | { ok: false; error: string };

function asSchema(value: unknown): JsonSchema {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonSchema)
    : {};
}

function number(value: unknown): number | undefined {
  return typeof value === "number" ? value : undefined;
}

/** "max_tokens" → "Max tokens". */
export function labelFor(name: string): string {
  const words = name.replace(/_/g, " ").trim();
  return words ? words[0]!.toUpperCase() + words.slice(1) : name;
}

/** A property's schema without its `null` alternative, and whether it had one. */
function unwrapNullable(schema: JsonSchema): {
  schema: JsonSchema;
  nullable: boolean;
} {
  const anyOf = schema.anyOf;
  if (Array.isArray(anyOf)) {
    const options = anyOf.map(asSchema);
    const rest = options.filter((option) => option.type !== "null");
    if (rest.length === 1 && rest.length < options.length) {
      const outer = { ...schema };
      delete outer.anyOf;
      return { schema: { ...outer, ...rest[0] }, nullable: true };
    }
  }
  return { schema, nullable: false };
}

function kindOf(schema: JsonSchema): FieldKind {
  if (Array.isArray(schema.enum) || "const" in schema) return "choice";
  switch (schema.type) {
    case "integer":
      return "integer";
    case "number":
      return "number";
    case "boolean":
      return "boolean";
    case "string":
      return "text";
    default:
      return "json";
  }
}

/** The fields of an object schema, in the order its properties appear. */
export function fieldsFromSchema(schema: JsonSchema): Field[] {
  const properties = asSchema(schema.properties);
  return Object.entries(properties).map(([name, raw]) => {
    const { schema: property, nullable } = unwrapNullable(asSchema(raw));
    const kind = kindOf(property);
    const field: Field = {
      name,
      label: labelFor(name),
      kind,
      nullable,
    };
    if (typeof property.description === "string") {
      field.description = property.description;
    }
    if ("default" in property) field.default = property.default;
    if (kind === "choice") {
      field.options = Array.isArray(property.enum)
        ? (property.enum as (string | number)[])
        : [property.const as string | number];
    }
    field.minimum = number(property.minimum);
    field.maximum = number(property.maximum);
    field.exclusiveMinimum = number(property.exclusiveMinimum);
    field.exclusiveMaximum = number(property.exclusiveMaximum);
    field.maxLength = number(property.maxLength);
    if (typeof property.pattern === "string") field.pattern = property.pattern;
    return field;
  });
}

/** Every default an object schema declares: a component's parameters as the baseline has them. */
export function defaultsFromSchema(
  schema: JsonSchema,
): Record<string, unknown> {
  const defaults: Record<string, unknown> = {};
  for (const field of fieldsFromSchema(schema)) {
    if (field.default !== undefined) defaults[field.name] = field.default;
  }
  return defaults;
}

function checkRange(field: Field, value: number): string | null {
  if (field.minimum !== undefined && value < field.minimum) {
    return `At least ${field.minimum}`;
  }
  if (field.maximum !== undefined && value > field.maximum) {
    return `At most ${field.maximum}`;
  }
  if (field.exclusiveMinimum !== undefined && value <= field.exclusiveMinimum) {
    return `More than ${field.exclusiveMinimum}`;
  }
  if (field.exclusiveMaximum !== undefined && value >= field.exclusiveMaximum) {
    return `Less than ${field.exclusiveMaximum}`;
  }
  return null;
}

/**
 * A form input's text as the field's value, or why it isn't one. Empty input
 * is null for a nullable field and the default otherwise.
 */
export function coerce(field: Field, raw: string): Coerced {
  const text = raw.trim();
  if (text === "") {
    if (field.nullable) return { ok: true, value: null };
    if (field.kind === "text") return { ok: true, value: "" };
    if (field.default !== undefined) return { ok: true, value: field.default };
    return { ok: false, error: "Required" };
  }
  switch (field.kind) {
    case "integer": {
      if (!/^[+-]?\d+$/.test(text))
        return { ok: false, error: "A whole number" };
      const value = Number(text);
      const error = checkRange(field, value);
      return error ? { ok: false, error } : { ok: true, value };
    }
    case "number": {
      const value = Number(text);
      if (!Number.isFinite(value)) return { ok: false, error: "A number" };
      const error = checkRange(field, value);
      return error ? { ok: false, error } : { ok: true, value };
    }
    case "boolean":
      if (text === "true" || text === "false") {
        return { ok: true, value: text === "true" };
      }
      return { ok: false, error: "true or false" };
    case "choice": {
      const option = field.options?.find((choice) => String(choice) === text);
      return option === undefined
        ? { ok: false, error: "One of the choices" }
        : { ok: true, value: option };
    }
    case "text": {
      if (field.maxLength !== undefined && raw.length > field.maxLength) {
        return { ok: false, error: `At most ${field.maxLength} characters` };
      }
      if (field.pattern !== undefined && !new RegExp(field.pattern).test(raw)) {
        return { ok: false, error: "Not in the expected form" };
      }
      return { ok: true, value: raw };
    }
    case "json":
      try {
        return { ok: true, value: JSON.parse(text) };
      } catch {
        return { ok: false, error: "Valid JSON" };
      }
  }
}

/** A value as a form input's text. */
export function display(field: Field, value: unknown): string {
  if (value === undefined || value === null) return "";
  if (field.kind === "json") return JSON.stringify(value);
  return String(value);
}
