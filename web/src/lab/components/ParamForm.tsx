import { useId, useState } from "react";
import {
  coerce,
  display,
  fieldsFromSchema,
  type Field,
  type JsonSchema,
} from "../spec/schemaForm";
import { inputClass } from "./ui";

/**
 * A component's parameters as a form generated from its JSON Schema.
 * Each change that checks out goes to `onChange`; one that doesn't shows why
 * and changes nothing.
 */
export function ParamForm({
  schema,
  values,
  onChange,
  idPrefix,
}: {
  schema: JsonSchema;
  values: Record<string, unknown>;
  onChange: (name: string, value: unknown) => void;
  idPrefix: string;
}) {
  const fields = fieldsFromSchema(schema);
  if (fields.length === 0) {
    return <p className="text-xs text-muted">No parameters.</p>;
  }
  return (
    <div className="grid gap-2">
      {fields.map((field) => (
        <FieldInput
          key={`${idPrefix}.${field.name}:${display(field, values[field.name] ?? field.default)}`}
          field={field}
          value={values[field.name] ?? field.default}
          onChange={(value) => onChange(field.name, value)}
        />
      ))}
    </div>
  );
}

function FieldInput({
  field,
  value,
  onChange,
}: {
  field: Field;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const id = useId();
  const [text, setText] = useState(display(field, value));
  const [error, setError] = useState<string | null>(null);
  const commit = (raw: string) => {
    const result = coerce(field, raw);
    if (result.ok) {
      setError(null);
      onChange(result.value);
    } else {
      setError(result.error);
    }
  };
  const describedBy = field.description || error ? `${id}-help` : undefined;
  let control;
  if (field.kind === "boolean") {
    control = (
      <input
        id={id}
        type="checkbox"
        checked={value === true}
        onChange={(event) => onChange(event.target.checked)}
        aria-describedby={describedBy}
      />
    );
  } else if (field.kind === "choice") {
    control = (
      <select
        id={id}
        className={inputClass}
        value={text}
        disabled={(field.options?.length ?? 0) < 2}
        onChange={(event) => {
          setText(event.target.value);
          commit(event.target.value);
        }}
        aria-describedby={describedBy}
      >
        {field.options?.map((option) => (
          <option key={String(option)} value={String(option)}>
            {String(option)}
          </option>
        ))}
      </select>
    );
  } else {
    control = (
      <input
        id={id}
        className={inputClass}
        type="text"
        inputMode={
          field.kind === "integer" || field.kind === "number"
            ? "decimal"
            : undefined
        }
        value={text}
        onChange={(event) => setText(event.target.value)}
        onBlur={(event) => commit(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") commit(event.currentTarget.value);
        }}
        aria-invalid={error ? true : undefined}
        aria-describedby={describedBy}
      />
    );
  }
  return (
    <div className="grid gap-1">
      <label htmlFor={id} className="text-xs font-medium">
        {field.label}
      </label>
      {control}
      {field.description || error ? (
        <p
          id={`${id}-help`}
          className={`text-xs ${error ? "text-danger" : "text-muted"}`}
        >
          {error ?? field.description}
        </p>
      ) : null}
    </div>
  );
}
