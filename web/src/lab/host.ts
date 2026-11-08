import type { ComponentType, ReactNode } from "react";

export type NotifyTone = "info" | "success" | "warning" | "danger";

export interface ConfirmRequest {
  title: string;
  body: string;
  confirmLabel: string;
  danger?: boolean;
}

export interface FrameProps {
  title: string;
  actions?: ReactNode;
  children: ReactNode;
}

/**
 * Everything the lab needs from the page around it. The lab depends
 * on nothing else: no global auth or routing, so the standalone shell is just
 * one host, and another page could embed the lab the same way.
 */
export interface StratafiHost {
  /** Where the lab is mounted: "/stratafi/" standalone, or wherever another host mounts it. */
  basePath: string;
  /** The API's base URL: "/api/v1". */
  apiBase: string;
  /** Headers that authenticate an API call, fresh for each call. */
  authHeaders(): Promise<Record<string, string>>;
  /** Go to a path inside the lab, relative to basePath. */
  navigate(path: string, options?: { replace?: boolean }): void;
  Frame: ComponentType<FrameProps>;
  notify(message: string, tone?: NotifyTone): void;
  confirm(request: ConfirmRequest): Promise<boolean>;
}
