import { Button, Card } from "@heroui/react";
import { useState, type FormEvent } from "react";
import {
  LOCAL_OWNER,
  SIGN_IN_FAILED,
  SignInUnavailable,
  type DevIdentity,
  type DevSession,
} from "./session";

const ROLES = ["owner", "editor", "viewer"];
const field =
  "rounded-md border border-separator bg-background px-2 py-1 text-sm";

/**
 * Development sign-in: pick a synthetic user; the API mints its short-lived
 * tokens. The lab signs in as the local owner by itself, so this form is for
 * switching user, or for saying why that sign-in failed.
 */
export function SignIn({
  session,
  onSignedIn,
  initialError = null,
}: {
  session: DevSession;
  onSignedIn: (identity: DevIdentity) => void;
  initialError?: string | null;
}) {
  const [tenant, setTenant] = useState(LOCAL_OWNER.tenant);
  const [role, setRole] = useState(LOCAL_OWNER.role);
  const [error, setError] = useState<string | null>(initialError);
  const [busy, setBusy] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    const identity = { tenant: tenant.trim(), role };
    try {
      await session.signIn(identity);
      onSignedIn(identity);
    } catch (failure) {
      setError(
        failure instanceof SignInUnavailable ? failure.message : SIGN_IN_FAILED,
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-6 text-foreground">
      <Card className="w-full max-w-sm">
        <Card.Header>
          <Card.Title>Stratafi</Card.Title>
          <Card.Description>
            Sign in as a synthetic user: the local owner, or someone in another
            tenant with another role.
          </Card.Description>
        </Card.Header>
        <Card.Content>
          <form onSubmit={submit} className="grid gap-3">
            <label className="grid gap-1 text-xs font-medium">
              Tenant
              <input
                className={field}
                value={tenant}
                onChange={(event) => setTenant(event.target.value)}
                pattern="[a-z0-9_]{1,64}"
                required
              />
            </label>
            <label className="grid gap-1 text-xs font-medium">
              Role
              <select
                className={field}
                value={role}
                onChange={(event) => setRole(event.target.value)}
              >
                {ROLES.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
            {error ? (
              <p role="alert" className="text-sm text-danger">
                {error}
              </p>
            ) : null}
            <Button type="submit" isDisabled={busy}>
              Sign in
            </Button>
          </form>
        </Card.Content>
      </Card>
    </div>
  );
}
