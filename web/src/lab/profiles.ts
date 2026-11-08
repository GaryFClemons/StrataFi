// Pure helpers for the profile screens: a
// profile's configuration as rows, and saving text the API returned as a file.

export interface ConfigRow {
  path: string;
  value: string;
}

function shown(value: unknown): string {
  if (value === null) return "null";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

/**
 * A pipeline spec or retrieval config as one row per setting, by path
 * (stages.chunk.params.max_tokens), in the document's order. Its schema and
 * name are left out: the profile names them.
 */
export function configRows(document: unknown, prefix = ""): ConfigRow[] {
  if (document === null || typeof document !== "object") {
    return prefix ? [{ path: prefix, value: shown(document) }] : [];
  }
  const entries = Array.isArray(document)
    ? document.map((value, index) => [String(index), value] as const)
    : Object.entries(document);
  if (entries.length === 0) {
    return prefix ? [{ path: prefix, value: shown(document) }] : [];
  }
  const rows: ConfigRow[] = [];
  for (const [key, value] of entries) {
    if (!prefix && (key === "schema" || key === "name")) continue;
    rows.push(...configRows(value, prefix ? `${prefix}.${key}` : key));
  }
  return rows;
}

/** Save text as a file, as the browser downloads one. */
export function downloadText(text: string, name: string, type: string): void {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
