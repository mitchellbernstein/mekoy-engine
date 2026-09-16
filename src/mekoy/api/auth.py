"""API-key auth for the control plane, and the one place a caller is resolved.

The provider is not specified, and Phase I runs locally. This is the smallest thing
that is actually auth rather than a placeholder — a bearer key, compared in constant
time, required on `/v1` when one is configured.

Off by default. An unset `MEKOY_API_KEY` and `MEKOY_API_KEYS` leaves the API open,
which is the right default for a laptop and the wrong one for anything reachable, so
the app logs a warning when it is serving `/v1` without one.

`/health` stays open: a liveness probe that needs a credential cannot be probed by
the thing checking liveness. `/mcp` is guarded alongside `/v1`: the MCP HTTP
transport carries an `Authorization` header, so a connector has somewhere to put the
key. Leaving it open was a gap, not a design — an unauthenticated endpoint that
accepts compile jobs is not something to put on the internet.

## Identity

Auth used to be one shared key with no notion of who called. That is fine for a
laptop and a breach the moment there is more than one customer, because a shared key
means every caller is the same caller. `Principal` is the single resolution point:
every route asks the same function who is calling, and no route invents its own
answer.

Two modes, chosen by configuration alone — no service, no fork:

- **Local** (`MEKOY_API_KEYS` and `MEKOY_API_KEY` both unset): keyless, exactly as
  before. The principal is `LOCAL` and the store does not filter, so a System built
  by an older version, or by hand, stays reachable. A self-hoster never has to learn
  what a principal is.
- **Hosted** (a key is configured): the presented key names a principal.
  `MEKOY_API_KEYS` is `id:key,id:key` for real tenancy; a lone `MEKOY_API_KEY` is a
  single-tenant key and still identifies one caller.

`MEKOY_API_KEYS` rather than a database of accounts because a self-hoster must stay
at one process and one file. A hosted deployment replaces this with a lookup; the
resolution point is the same function either way.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from fastapi import Request
    from starlette.middleware.base import RequestResponseEndpoint
    from starlette.responses import Response

__all__ = [
    "API_KEYS_ENV",
    "API_KEY_ENV",
    "LOCAL",
    "LOCAL_ID",
    "AuthMiddleware",
    "AuthSettings",
    "Principal",
    "resolve_principal",
    "settings_from_env",
]

API_KEY_ENV = "MEKOY_API_KEY"
#: `id:key,id:key`. The hosted, multi-tenant form. Preferred over `MEKOY_API_KEY`.
API_KEYS_ENV = "MEKOY_API_KEYS"
#: Paths that never require a key.
_OPEN_PREFIXES = ("/health", "/docs", "/openapi.json", "/redoc")
#: Paths that do. `/mcp` is included: the MCP HTTP transport accepts an
#: Authorization header, so an open connector endpoint was never necessary — it
#: was just unguarded. Left open, a deploy would expose an unauthenticated
#: endpoint that accepts compile jobs.
_GUARDED_PREFIXES = ("/v1", "/mcp")

#: The principal a keyless, local control plane runs as. Its id is what a System
#: created before this existed is recorded with, so those rows stay reachable.
LOCAL_ID = "local"


@dataclass(frozen=True, slots=True)
class Principal:
    """Who is calling, for the purposes of ownership and quota.

    `id` is the tenancy key: every row written while this principal is in scope is
    stamped with it, and every read is filtered by it. An empty `id` never reaches
    the store — local mode is expressed by the scope being unfiltered, not by an
    owner of `None`, so an unowned row is a bug rather than a shared-with-everyone
    row.
    """

    id: str
    #: The env key this principal came in on, for an operator reading a log. Empty
    #: in local mode, which is what `is_local` keys off.
    key_name: str = ""

    @property
    def is_local(self) -> bool:
        """Whether this is the keyless, single-tenant local principal."""
        return self.id == LOCAL_ID and not self.key_name


#: The one principal a keyless control plane has.
LOCAL = Principal(id=LOCAL_ID)


@dataclass(frozen=True, slots=True)
class AuthSettings:
    """Whether auth is on, and which principal each key names."""

    api_key: str = ""
    #: key -> principal id. Includes `api_key` when only the legacy var is set.
    keys: dict[str, str] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        """Auth is on exactly when a key is configured."""
        return bool(self.api_key) or bool(self.keys)

    def accepts(self, presented: str | None) -> bool:
        """Constant-time compare, so a wrong key cannot be guessed by timing."""
        if not self.enabled:
            return True
        return self.principal_for(presented) is not None

    def principal_for(self, presented: str | None) -> Principal | None:
        """The principal a presented key names, or None if it is not a real key.

        Compared against every configured key with `compare_digest`, and the whole
        set is scanned rather than short-circuiting, so which key matched is not
        readable from response timing. A bare `api_key` with no `keys` map is the
        single-tenant form and names `default`.
        """
        if not presented:
            return None
        table = self.keys or ({self.api_key: "default"} if self.api_key else {})
        found: Principal | None = None
        for key, owner in table.items():
            if secrets.compare_digest(key, presented):
                found = Principal(id=owner, key_name=owner)
        return found


def settings_from_env() -> AuthSettings:
    """Read keys from the environment. Empty means auth is off."""
    many = _parse_keys(os.environ.get(API_KEYS_ENV, ""))
    if many:
        return AuthSettings(keys=many)
    one = os.environ.get(API_KEY_ENV, "").strip()
    # A single key still names one caller (`default`), which is what makes the stored
    # rows filterable rather than ambiguous.
    return AuthSettings(api_key=one)


def _parse_keys(raw: str) -> dict[str, str]:
    """`tenant-a:key-a,tenant-b:key-b` -> {key: principal id}.

    A malformed entry is skipped rather than fatal: refusing to boot over one bad
    pair would take a running control plane down for a typo.
    """
    out: dict[str, str] = {}
    for part in raw.split(","):
        entry = part.strip()
        if not entry or ":" not in entry:
            continue
        owner, _, key = entry.partition(":")
        owner, key = owner.strip(), key.strip()
        if owner and key:
            out[key] = owner
    return out


def _presented(request: Request) -> str | None:
    """Pull the bearer token out of the Authorization header."""
    header = request.headers.get("authorization", "")
    prefix = "bearer "
    if header.lower().startswith(prefix):
        return header[len(prefix) :].strip()
    # A plain `X-API-Key` is friendlier for curl and for a browser fetch.
    return request.headers.get("x-api-key")


def _requires_auth(path: str) -> bool:
    if path.startswith(_OPEN_PREFIXES):
        return False
    return path.startswith(_GUARDED_PREFIXES)


def resolve_principal(request: Request) -> Principal:
    """The one place a caller is identified.

    Middleware stashes the result on `request.state`; this reads it back and never
    re-derives it, so HTTP routes and anything holding the request agree. A request
    that never passed through the middleware — a test client, an ASGI mount — gets
    the local principal, which is the same answer local mode gives.
    """
    principal = getattr(request.state, "principal", None)
    return principal if isinstance(principal, Principal) else LOCAL


class AuthMiddleware(BaseHTTPMiddleware):
    """Reject unkeyed `/v1` requests and record who is calling."""

    def __init__(self, app: object, *, settings: AuthSettings) -> None:
        """Wrap an ASGI app with the given auth settings."""
        super().__init__(app)  # type: ignore[arg-type]
        self._settings = settings

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Reject an unkeyed `/v1` request when a key is configured.

        Sets `request.state.principal` on the way through, so a route resolves the
        caller from the same place regardless of which path it was reached by.
        """
        presented = _presented(request)
        principal = (
            self._settings.principal_for(presented) if self._settings.enabled else LOCAL
        )
        guarded = self._settings.enabled and _requires_auth(request.url.path)
        if guarded and principal is None:
            return JSONResponse(
                {"detail": "missing or invalid API key"}, status_code=401
            )
        request.state.principal = principal or LOCAL
        return await call_next(request)
