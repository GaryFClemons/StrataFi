import {
  createContext,
  useContext,
  useSyncExternalStore,
  type MouseEvent,
  type ReactNode,
} from "react";
import type { ApiClient } from "./api/client";
import type { StratafiHost } from "./host";
import { toBrowserPath, toLabPath } from "./paths";
import { routePath, type Route } from "./routes";

export interface Lab {
  host: StratafiHost;
  api: ApiClient;
}

export const LabContext = createContext<Lab | null>(null);

export function useLab(): Lab {
  const lab = useContext(LabContext);
  if (lab === null) throw new Error("useLab outside <StratafiLab>");
  return lab;
}

function subscribe(onChange: () => void): () => void {
  window.addEventListener("popstate", onChange);
  return () => window.removeEventListener("popstate", onChange);
}

/** The current lab path; hosts announce navigation with popstate. */
export function useLabPath(basePath: string): string {
  return useSyncExternalStore(
    subscribe,
    () => toLabPath(basePath, window.location.pathname) ?? "/",
    () => "/",
  );
}

export function useNavigate(): (
  route: Route,
  options?: { replace?: boolean },
) => void {
  const { host } = useLab();
  return (route, options) => host.navigate(routePath(route), options);
}

/** A link inside the lab: a real href, with in-app navigation on a plain click. */
export function Link({
  to,
  children,
  className,
  current,
}: {
  to: Route;
  children: ReactNode;
  className?: string;
  current?: boolean;
}) {
  const { host } = useLab();
  const path = routePath(to);
  const onClick = (event: MouseEvent<HTMLAnchorElement>) => {
    if (
      event.button !== 0 ||
      event.metaKey ||
      event.ctrlKey ||
      event.shiftKey ||
      event.altKey
    ) {
      return;
    }
    event.preventDefault();
    host.navigate(path);
  };
  return (
    <a
      href={toBrowserPath(host.basePath, path)}
      onClick={onClick}
      className={className}
      aria-current={current ? "page" : undefined}
    >
      {children}
    </a>
  );
}
