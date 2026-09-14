# Install Mekoy on every host

The same tools. Two transports.

| Who | How |
|---|---|
| **Local hardware** (Ollama, Apple Silicon, GPU) | stdio MCP. Claude Code, Cursor, Grok Build, Codex CLI. |
| **No hardware** | HTTPS Streamable HTTP MCP. Claude.ai connector, ChatGPT plugin, Grok custom MCP, Cursor plugin with URL. |

Public URL: `https://mcp.mekoy.com/mcp`

Until that DNS is live, tunnel the local API:

```bash
uv run uvicorn mekoy.api.main:app --host 127.0.0.1 --port 8787
ngrok http 8787
# then use https://<id>.ngrok.app/mcp
```

## Claude Code (local)

```bash
claude mcp add --scope user mekoy -- uv run --directory /path/to/mekoy python -m mekoy.mcp_server
```

Type `/mcp` (not `/mcp mekoy`). Enable Mekoy. New session.

## Claude.ai connector (remote)

Customize → Connectors → Add custom connector → Web.

MCP server URL: `https://mcp.mekoy.com/mcp` (or your ngrok URL + `/mcp`).

No sign-in for v1.

## ChatGPT / Codex plugin

1. Enable developer mode.
2. Settings → Plugins → create app **or** add custom MCP.
3. MCP server URL: `https://mcp.mekoy.com/mcp`
4. Or install the plugin package in `connectors/chatgpt/` via a marketplace once published.

Codex CLI: `/plugins` after the plugin is in a marketplace. Same MCP URL.

## Cursor plugin

Repo: `connectors/` with `.cursor-plugin/plugin.json`.

Or Settings → MCP → URL `https://mcp.mekoy.com/mcp`.

Local stdio still works via `~/.cursor/mcp.json` command `uv run python -m mekoy.mcp_server`.

## Grok Build / Grok web / Grok Bot

```bash
grok mcp add --transport http mekoy https://mcp.mekoy.com/mcp
```

Grok web/iOS: add a custom MCP connector with that HTTPS URL (localhost is rejected; use a tunnel).

Plugin marketplace: submit `connectors/grok/` to `xai-org/plugin-marketplace` when ready.

Grok Bot: ship `connectors/skills/compile/SKILL.md` as a bot skill plus the same MCP.

## This repo's portal chat

`/app/systems/new` is the hosted twin of these connectors. Same tools, same eval gate.
