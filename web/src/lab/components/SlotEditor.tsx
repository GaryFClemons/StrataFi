// One component slot of the pipeline spec: a component picker and the form
// its parameter schema generates. Every step screen edits slots.

import type { ReactNode } from "react";
import type { components } from "../api";
import { ParamForm } from "./ParamForm";
import { inputClass } from "./ui";
import {
  getAt,
  refAt,
  setAt,
  withComponent,
  withParam,
  type Slot,
  type SpecDocument,
} from "../spec/editing";
import { defaultsFromSchema, type JsonSchema } from "../spec/schemaForm";

type ComponentOut = components["schemas"]["ComponentOut"];

export function SlotEditor({
  slot,
  document,
  components,
  onChange,
  actions,
}: {
  slot: Slot;
  document: SpecDocument;
  components: ComponentOut[];
  onChange: (document: SpecDocument) => void;
  actions?: ReactNode;
}) {
  const ref = refAt(document, slot.path);
  if (ref === null) return null;
  const choices = components.filter(
    (component) =>
      component.stage === slot.stage &&
      (!slot.prefix || component.id.startsWith(slot.prefix)) &&
      (!slot.only || slot.only.includes(component.id)),
  );
  const current = components.find(
    (component) => component.id === ref.component,
  );
  const selectId = `component-${slot.path}`;
  return (
    <fieldset className="grid gap-2 rounded-md border border-separator p-3">
      <legend className="px-1 text-xs font-semibold">{slot.label}</legend>
      {actions ? <div className="flex flex-wrap gap-1">{actions}</div> : null}
      {slot.fixed ? (
        <p className="text-sm">{current?.title ?? ref.component}</p>
      ) : (
        <div className="grid gap-1">
          <label htmlFor={selectId} className="text-xs font-medium">
            Component
          </label>
          <select
            id={selectId}
            className={inputClass}
            value={ref.component}
            onChange={(event) => {
              const chosen = components.find(
                (component) => component.id === event.target.value,
              );
              if (!chosen) return;
              onChange(
                withComponent(
                  document,
                  slot.path,
                  chosen.id,
                  defaultsFromSchema(chosen.params_schema as JsonSchema),
                ),
              );
            }}
          >
            {choices.map((component) => (
              <option key={component.id} value={component.id}>
                {component.title}
                {component.baseline ? " (baseline)" : ""}
                {component.available ? "" : " (not installed)"}
              </option>
            ))}
          </select>
        </div>
      )}
      {current?.description ? (
        <p className="text-xs text-muted">{current.description}</p>
      ) : null}
      {current && !current.available ? (
        <p className="text-xs text-warning" role="note">
          This installation can&apos;t run it: {current.unavailable_reason}
        </p>
      ) : null}
      {current ? (
        <ParamForm
          idPrefix={slot.path}
          schema={current.params_schema as JsonSchema}
          values={ref.params}
          onChange={(name, value) =>
            onChange(withParam(document, slot.path, name, value))
          }
        />
      ) : null}
      {slot.path === "stages.chunk" ? (
        <label className="grid gap-1 text-xs font-medium">
          Index text
          <input
            className={inputClass}
            value={String(getAt(document, "stages.chunk.index_text") ?? "")}
            onChange={(event) =>
              onChange(
                setAt(document, "stages.chunk.index_text", event.target.value),
              )
            }
          />
          <span className="font-normal text-muted">
            What is embedded and keyword-indexed: {"{title}"},{" "}
            {"{heading_path}"}, {"{text}"}
          </span>
        </label>
      ) : null}
    </fieldset>
  );
}
