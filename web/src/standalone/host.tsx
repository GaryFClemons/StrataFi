import { Button, toast } from "@heroui/react";
import type { FrameProps, NotifyTone, StratafiHost } from "../lab";
import { normalizeBasePath, toBrowserPath } from "../lab";
import type { StandaloneSession } from "./session";

/** Where the standalone app is served: Vite's base (STRATAFI_BASE_PATH at build time). */
export const STANDALONE_BASE_PATH = import.meta.env.BASE_URL || "/stratafi/";

function frameFor(session: StandaloneSession, onSwitchUser: () => void) {
  return function StandaloneFrame({ title, actions, children }: FrameProps) {
    const label = session.label();
    return (
      <div className="min-h-screen bg-background text-foreground">
        <header className="flex items-center justify-between gap-4 border-b border-separator px-6 py-3">
          <h1 className="text-lg font-semibold">{title}</h1>
          <div className="flex items-center gap-2">
            {actions}
            {label ? (
              <span className="text-xs text-muted" data-testid="signed-in-as">
                {label}
              </span>
            ) : null}
            <Button size="sm" variant="tertiary" onPress={onSwitchUser}>
              Switch user
            </Button>
          </div>
        </header>
        <main className="mx-auto max-w-7xl p-6">{children}</main>
      </div>
    );
  };
}

const toastByTone: Record<NotifyTone, (message: string) => string> = {
  info: (message) => toast.info(message),
  success: (message) => toast.success(message),
  warning: (message) => toast.warning(message),
  danger: (message) => toast.danger(message),
};

/**
 * The lab's standalone host: it signs API calls with the development
 * sign-in's identity tokens, as the local owner unless you switch
 * user.
 */
export function createStandaloneHost(
  session: StandaloneSession,
  onSwitchUser: () => void,
  basePath: string = STANDALONE_BASE_PATH,
): StratafiHost {
  const base = normalizeBasePath(basePath);
  return {
    basePath: base,
    apiBase: "/api/v1",
    authHeaders: () => session.headers(),
    navigate(path, options) {
      const url = toBrowserPath(base, path);
      if (options?.replace) window.history.replaceState(null, "", url);
      else window.history.pushState(null, "", url);
      window.dispatchEvent(new PopStateEvent("popstate"));
    },
    Frame: frameFor(session, onSwitchUser),
    notify(message, tone = "info") {
      toastByTone[tone](message);
    },
    confirm: async ({ title, body }) => window.confirm(`${title}\n\n${body}`),
  };
}
