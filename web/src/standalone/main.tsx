import { Toast } from "@heroui/react";
import { StrictMode, useMemo, useState, type ReactElement } from "react";
import { createRoot } from "react-dom/client";
import { StratafiLab } from "../lab";
import { createStandaloneHost } from "./host";
import { DevSession, signInLocally } from "./session";
import { SignIn } from "./SignIn";
import "./styles.css";

const API_BASE = "/api/v1";

function sessionStorageOrNull(): Storage | null {
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

function Standalone({
  session,
  bootError,
}: {
  session: DevSession;
  bootError: string | null;
}) {
  const [signedIn, setSignedIn] = useState(session.label() !== null);
  const [error, setError] = useState(bootError);
  const host = useMemo(
    () =>
      createStandaloneHost(session, () => {
        session.signOut();
        setSignedIn(false);
      }),
    [session],
  );
  if (!signedIn)
    return (
      <SignIn
        session={session}
        onSignedIn={() => {
          setError(null);
          setSignedIn(true);
        }}
        initialError={error}
      />
    );
  return <StratafiLab host={host} />;
}

/**
 * The lab signs in as the local owner by itself; the sign-in form shows only
 * to switch user, or to say why that sign-in failed.
 */
async function boot(): Promise<ReactElement> {
  const session = new DevSession(API_BASE, sessionStorageOrNull());
  const error = await signInLocally(session, API_BASE);
  return <Standalone session={session} bootError={error} />;
}

const container = document.getElementById("root");
if (!container) throw new Error("index.html has no #root element");

void boot().then((app) =>
  createRoot(container).render(
    <StrictMode>
      <Toast.Provider />
      {app}
    </StrictMode>,
  ),
);
