// The standalone host's development sign-in. Identity tokens live at
// most five minutes, so the session keeps who you signed in as and fetches a
// fresh token from the API's development route as the current one ages.
// Tokens stay in memory; only the tenant and role are kept for the tab.

/** What the standalone host needs from a sign-in. */
export interface StandaloneSession {
  headers(): Promise<Record<string, string>>;
  signOut(): void;
  /** Who is signed in, for the frame; null when signed out. */
  label(): string | null;
}

export interface DevIdentity {
  tenant: string;
  role: string;
}

/** Who the lab signs in as by itself: the one person on this machine. */
export const LOCAL_OWNER: DevIdentity = { tenant: "local", role: "owner" };

export const SIGN_IN_FAILED = "Sign-in failed. Is the API up?";

interface TokenResponse {
  token: string;
  expires_in: number;
  subject: string;
}

const STORAGE_KEY = "stratafi.dev-identity";
/** Refresh this long before a token expires. */
const MARGIN_SECONDS = 60;

export class SignInUnavailable extends Error {
  constructor() {
    super(
      "Development sign-in is off. Set STRATAFI_DEV_SIGN_IN_KEY (the development key) and restart the API.",
    );
  }
}

export class DevSession implements StandaloneSession {
  private token: { value: string; expires: number } | null = null;
  private pending: Promise<string> | null = null;

  constructor(
    private readonly apiBase: string,
    private readonly storage: Pick<
      Storage,
      "getItem" | "setItem" | "removeItem"
    > | null,
    private readonly fetchImpl: typeof fetch = (...args) => fetch(...args),
    private readonly now: () => number = () => Date.now(),
  ) {}

  identity(): DevIdentity | null {
    try {
      const stored = this.storage?.getItem(STORAGE_KEY);
      const parsed: unknown = stored ? JSON.parse(stored) : null;
      if (
        parsed !== null &&
        typeof parsed === "object" &&
        typeof (parsed as DevIdentity).tenant === "string" &&
        typeof (parsed as DevIdentity).role === "string"
      ) {
        return parsed as DevIdentity;
      }
    } catch {
      // Storage unavailable or corrupt: signed out.
    }
    return null;
  }

  /** Check the identity works, then keep it for the tab. */
  async signIn(identity: DevIdentity): Promise<void> {
    this.token = null;
    await this.fetchToken(identity);
    try {
      this.storage?.setItem(STORAGE_KEY, JSON.stringify(identity));
    } catch {
      // Not kept: signed in until the page reloads.
      this.fallback = identity;
    }
  }

  private fallback: DevIdentity | null = null;

  label(): string | null {
    const identity = this.identity() ?? this.fallback;
    return identity
      ? `${identity.role} in ${identity.tenant} (development)`
      : null;
  }

  signOut(): void {
    this.token = null;
    this.fallback = null;
    try {
      this.storage?.removeItem(STORAGE_KEY);
    } catch {
      // Nothing kept.
    }
  }

  /** The identity header for the next call, or none when signed out. */
  async headers(): Promise<Record<string, string>> {
    const identity = this.identity() ?? this.fallback;
    if (identity === null) return {};
    if (this.token === null || this.now() >= this.token.expires) {
      this.pending ??= this.fetchToken(identity).finally(() => {
        this.pending = null;
      });
      await this.pending;
    }
    return this.token ? { "X-Stratafi-Identity": this.token.value } : {};
  }

  private async fetchToken(identity: DevIdentity): Promise<string> {
    const response = await this.fetchImpl(
      `${this.apiBase.replace(/\/+$/, "")}/dev/token`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(identity),
      },
    );
    if (response.status === 404) throw new SignInUnavailable();
    if (!response.ok) throw new Error(`Sign-in failed (${response.status}).`);
    const body = (await response.json()) as TokenResponse;
    const lifetime = Math.max(body.expires_in - MARGIN_SECONDS, 1) * 1000;
    this.token = { value: body.token, expires: this.now() + lifetime };
    return body.token;
  }
}

/**
 * Sign in as the local owner when the tab has no one yet and the API's
 * development sign-in is on (GET /auth/config). Returns why it couldn't, for
 * the sign-in form, or null.
 */
export async function signInLocally(
  session: DevSession,
  apiBase: string,
  fetchImpl: typeof fetch = (...args) => fetch(...args),
): Promise<string | null> {
  if (session.identity() !== null) return null;
  let on: boolean;
  try {
    const response = await fetchImpl(
      `${apiBase.replace(/\/+$/, "")}/auth/config`,
    );
    if (!response.ok) return SIGN_IN_FAILED;
    const config = (await response.json()) as { dev_sign_in?: unknown };
    on = config.dev_sign_in === true;
  } catch {
    return SIGN_IN_FAILED;
  }
  if (!on) return new SignInUnavailable().message;
  try {
    await session.signIn(LOCAL_OWNER);
    return null;
  } catch (failure) {
    return failure instanceof SignInUnavailable
      ? failure.message
      : SIGN_IN_FAILED;
  }
}
