"""Streamable HTTP MCP for Claude.ai, ChatGPT, Grok, Cursor, and Codex."""

from __future__ import annotations

import json
from collections.abc import Callable

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import TypeAdapter, ValidationError

from mekoy.mcp_server import Server

_JSON: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])


#: Set by the API when it mounts this router. A callable rather than a dependency:
#: the API imports this module, so this module cannot import the API to ask for its
#: context, and a `Depends` on an opaque type breaks the generated schema.
class _Binding:
    """Where the router keeps the way back to its app's context."""

    provider: Callable[[], object] | None = None


_context = _Binding()


def bind_context(provider: Callable[[], object]) -> None:
    """Tell this router how to reach the app's context, once, at mount time."""
    _context.provider = provider


router = APIRouter()
_server = Server()


@router.options("/mcp")
def mcp_options() -> Response:
    """CORS preflight for browser hosts."""
    return Response(status_code=204)


@router.get("/mcp")
def mcp_get() -> PlainTextResponse:
    """Some hosts probe GET. Streamable HTTP is POST-only here."""
    return PlainTextResponse(
        "Mekoy MCP. POST JSON-RPC to this URL.",
        status_code=200,
    )


@router.post("/mcp")
async def mcp_post(request: Request) -> Response:
    """JSON-RPC initialize, tools/list, tools/call."""
    try:
        payload = _JSON.validate_python(await request.json())
    except (ValidationError, ValueError, json.JSONDecodeError):
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            },
            status_code=400,
        )
    # The store is bound per request rather than held on the module-level server, so a
    # System compiled here lands where the HTTP API can find it and no request inherits
    # another app's state.
    bound = _context.provider
    _server.sink = getattr(bound(), "store", None) if bound is not None else None
    got = _server.handle(payload)
    if got is None:
        return Response(status_code=202)
    accept = request.headers.get("accept", "")
    if "text/event-stream" in accept and "application/json" not in accept:
        body = f"event: message\ndata: {json.dumps(got)}\n\n"
        return Response(content=body, media_type="text/event-stream")
    return JSONResponse(got)
