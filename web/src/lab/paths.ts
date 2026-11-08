// Base-path arithmetic for the host adapter. Every URL the lab makes is
// relative to host.basePath, so it works at /stratafi/ standalone and
// wherever a host mounts it.

/** "/stratafi" or "stratafi/" → "/stratafi/"; "" or "/" → "/". */
export function normalizeBasePath(basePath: string): string {
  const trimmed = basePath.trim().replace(/^\/+|\/+$/g, "");
  return trimmed ? `/${trimmed}/` : "/";
}

/** The browser path for a lab path: ("/stratafi/", "/runs/7") → "/stratafi/runs/7". */
export function toBrowserPath(basePath: string, labPath: string): string {
  if (/^[a-z][a-z0-9+.-]*:/i.test(labPath) || labPath.startsWith("//")) {
    throw new Error(
      "lab paths are relative to the base path, not absolute URLs",
    );
  }
  const base = normalizeBasePath(basePath);
  const path = labPath.replace(/^\/+/, "");
  return base + path;
}

/**
 * The lab path for a browser path, or null when the browser path is outside
 * the lab: ("/stratafi/", "/stratafi/runs/7") → "/runs/7".
 */
export function toLabPath(
  basePath: string,
  browserPath: string,
): string | null {
  const base = normalizeBasePath(basePath);
  if (browserPath === base.slice(0, -1) || browserPath === base) return "/";
  if (!browserPath.startsWith(base)) return null;
  return `/${browserPath.slice(base.length)}`;
}
