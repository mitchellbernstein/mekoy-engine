"""Streamable HTTP MCP for Claude.ai, ChatGPT, Grok, Cursor, and Codex."""

from __future__ import annotations

import json

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import TypeAdapter, ValidationError

from mekoy.mcp_server import Server

_JSON: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])

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
    got = _server.handle(payload)
    if got is None:
        return Response(status_code=202)
    accept = request.headers.get("accept", "")
    if "text/event-stream" in accept and "application/json" not in accept:
        body = f"event: message\ndata: {json.dumps(got)}\n\n"
        return Response(content=body, media_type="text/event-stream")
    return JSONResponse(got)
