"""API-key auth for the control plane.

PLAN §41.17 asks for "Auth (Clerk is fine) + object storage for artifacts". The
parenthesis matters: the provider is not specified, and Phase I runs locally. This
is the smallest thing that is actually auth rather than a placeholder — a bearer
key, compared in constant time, required on `/v1` when one is configured.

Off by default. An unset `MEKOY_API_KEY` leaves the API open, which is the right
default for a laptop and the wrong one for anything reachable, so the app logs a
warning when it is serving `/v1` without one.

`/health` stays open: a liveness probe that needs a credential cannot be probed by
the thing checking liveness. `/mcp` is guarded alongside `/v1`: the MCP HTTP
transport carries an `Authorization` header, so a connector has somewhere to put the
key. Leaving it open was a gap, not a design — an unauthenticated endpoint that
accepts compile jobs is not something to put on the internet.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from fastapi import Request
    from starlette.middleware.base import RequestResponseEndpoint
    from starlette.responses import Response

__all__ = ["API_KEY_ENV", "AuthMiddleware", "AuthSettings", "settings_from_env"]

API_KEY_ENV = "MEKOY_API_KEY"
#: Paths that never require a key.
_OPEN_PREFIXES = ("/health", "/docs", "/openapi.json", "/redoc")
#: Paths that do. `/mcp` is included: the MCP HTTP transport accepts an
#: Authorization header, so an open connector endpoint was never necessary — it
#: was just unguarded. Left open, a deploy would expose an unauthenticated
#: endpoint that accepts compile jobs.
_GUARDED_PREFIXES = ("/v1", "/mcp")


@dataclass(frozen=True, slots=True)
class AuthSettings:
    """Whether auth is on, and with which key."""

    api_key: str = ""

    @property
    def enabled(self) -> bool:
        """Auth is on exactly when a key is configured."""
        return bool(self.api_key)

    def accepts(self, presented: str | None) -> bool:
        """Constant-time compare, so a wrong key cannot be guessed by timing."""
        if not self.enabled:
            return True
        if not presented:
            return False
        return secrets.compare_digest(self.api_key, presented)


def settings_from_env() -> AuthSettings:
    """Read the key from the environment. Empty means auth is off."""
    return AuthSettings(api_key=os.environ.get(API_KEY_ENV, "").strip())


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


class AuthMiddleware(BaseHTTPMiddleware):
    """Reject unkeyed `/v1` requests when a key is configured."""

    def __init__(self, app: object, *, settings: AuthSettings) -> None:
        """Wrap an ASGI app with the given auth settings."""
        super().__init__(app)  # type: ignore[arg-type]
        self._settings = settings

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Reject an unkeyed `/v1` request when a key is configured."""
        guarded = self._settings.enabled and _requires_auth(request.url.path)
        if guarded and not self._settings.accepts(_presented(request)):
            return JSONResponse(
                {"detail": "missing or invalid API key"}, status_code=401
            )
        return await call_next(request)
