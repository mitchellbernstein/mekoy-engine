"""Production httpx2 client factory."""

from __future__ import annotations

import os
import socket
from collections.abc import Mapping

import httpx2

_LIMITS = httpx2.Limits(
    max_connections=200,
    max_keepalive_connections=40,
    keepalive_expiry=30.0,
)
#: A local model writing a long structured answer can take minutes. The old
#: 120-second read timeout turned that into "model server unreachable", which reads
#: as a connectivity problem when it is really a slow generation. Overridable
#: because a fast dedicated box wants a short timeout and a laptop wants a long one.
_READ_TIMEOUT_S = float(os.environ.get("MEKOY_READ_TIMEOUT_S", "600"))
_TIMEOUT = httpx2.Timeout(connect=5.0, read=_READ_TIMEOUT_S, write=10.0, pool=10.0)
_SOCKET_OPTIONS: list[tuple[int, int, int]] = [
    (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),
]


def create_client(
    *,
    base_url: str = "",
    headers: Mapping[str, str] | None = None,
) -> httpx2.Client:
    """Sync client with HTTP/2, retries, and LLM-friendly read timeout."""
    transport = httpx2.HTTPTransport(
        http2=True,
        retries=3,
        limits=_LIMITS,
        socket_options=_SOCKET_OPTIONS,
    )
    return httpx2.Client(
        transport=transport,
        timeout=_TIMEOUT,
        base_url=base_url,
        headers=dict(headers or {}),
        follow_redirects=True,
    )
