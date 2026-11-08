import { Card, Chip, Spinner } from "@heroui/react";
import type { ReactNode } from "react";
import { errorMessage } from "../api/load";
import { toneOf } from "../format";

export function Section({
  title,
  description,
  actions,
  children,
}: {
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <Card className="w-full">
      <Card.Header className="flex flex-row flex-wrap items-start justify-between gap-2">
        <div>
          <Card.Title>{title}</Card.Title>
          {description ? (
            <Card.Description>{description}</Card.Description>
          ) : null}
        </div>
        {actions ? (
          <div className="flex flex-wrap items-center gap-2">{actions}</div>
        ) : null}
      </Card.Header>
      <Card.Content>{children}</Card.Content>
    </Card>
  );
}

export function StatusChip({
  status,
  label,
}: {
  status: string;
  label?: string;
}) {
  return (
    <Chip size="sm" color={toneOf(status)} data-status={status}>
      {label ?? status.replace(/_/g, " ")}
    </Chip>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  if (!error) return null;
  return (
    <p role="alert" className="text-sm text-danger">
      {errorMessage(error)}
    </p>
  );
}

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-muted">
      <Spinner size="sm" /> {label}…
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="text-sm text-muted">{children}</p>;
}

export interface Column<T> {
  header: string;
  cell: (row: T) => ReactNode;
  className?: string;
}

/** A plain table: the API pages, filters and sorts; this only shows a page. */
export function DataTable<T>({
  rows,
  columns,
  rowKey,
  label,
  onSelect,
  selected,
}: {
  rows: T[];
  columns: Column<T>[];
  rowKey: (row: T) => string;
  label: string;
  onSelect?: (row: T) => void;
  selected?: string | null;
}) {
  return (
    <div className="w-full overflow-x-auto">
      <table
        aria-label={label}
        className="w-full border-collapse text-left text-sm"
      >
        <thead>
          <tr className="border-b border-separator">
            {columns.map((column) => (
              <th
                key={column.header}
                scope="col"
                className="px-2 py-1 font-medium text-muted"
              >
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const key = rowKey(row);
            const isSelected = selected === key;
            return (
              <tr
                key={key}
                className={`border-b border-separator ${
                  onSelect ? "cursor-pointer hover:bg-default" : ""
                } ${isSelected ? "bg-default" : ""}`}
                onClick={onSelect ? () => onSelect(row) : undefined}
                aria-selected={onSelect ? isSelected : undefined}
              >
                {columns.map((column) => (
                  <td
                    key={column.header}
                    className={`px-2 py-1 align-top ${column.className ?? ""}`}
                  >
                    {column.cell(row)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** Document or chunk text: always plain text, never markup. */
export function PlainText({ text, label }: { text: string; label: string }) {
  return (
    <pre
      aria-label={label}
      className="max-h-[32rem] overflow-auto whitespace-pre-wrap break-words rounded-md border border-separator p-3 font-mono text-xs"
    >
      {text}
    </pre>
  );
}

export const inputClass =
  "rounded-md border border-separator bg-background px-2 py-1 text-sm text-foreground";
